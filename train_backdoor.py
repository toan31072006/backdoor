import os, argparse, random, yaml, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.feeder import PersonInWiFi3D
from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
from attack.tsba import build_tsba_trigger
from attack.poison import PoisonedDataset, collate
from models.factory import build_model
from eval import metrics as M


# ── Checkpoint helpers ───────────────────────────────────────────────────────
# A 4x4 sweep at 200 epochs is ~24 h, so a dropped Colab session must not cost
# the whole grid. Each cell keeps its own checkpoint, keyed by a fingerprint of
# every result-affecting option: resuming a checkpoint trained under a different
# pivot / theta / rho / eps would silently report the wrong model's metrics.

_RESUME_SAFE_CFG_KEYS = {'device', 'epochs', 'ckpt_every', 'num_workers'}
_CHECKPOINT_SCHEMA = 6  # fully resolved defaults + DT-Pose train filtering


def _config_fingerprint(cfg):
    """Hash every result-affecting option while allowing runtime-only changes."""
    stable = {k: v for k, v in cfg.items() if k not in _RESUME_SAFE_CFG_KEYS}
    payload = json.dumps({'schema': _CHECKPOINT_SCHEMA, 'config': stable},
                         sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _ckpt_path(cfg, ckpt_dir=None):
    if ckpt_dir is None:
        return None
    base = Path(ckpt_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / 'checkpoint.pt'


def _rng_state():
    """Capture every RNG the training loop consumes.

    The DataLoader shuffles, so without these the resumed run replays the batch
    order from epoch 0 instead of continuing the original stream, and diverges
    from an uninterrupted run.

    NOTE: this covers the *global* torch RNG, which is what our DataLoader uses.
    If anyone later passes DataLoader(generator=g), that generator carries its
    own state and must be checkpointed too (g.get_state() / g.set_state()) —
    otherwise resume silently diverges again (measured ~3e-3 parameter drift).
    """
    return {'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)}


def _restore_rng_state(st):
    if not st:
        print('[ckpt] no RNG state stored; batch order will not match an '
              'uninterrupted run', flush=True)
        return
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu() if hasattr(st['torch'], 'cpu')
                        else st['torch'])
    if st.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def _save_checkpoint(path, model, optimizer, epoch, cfg,
                     trigger=None, trigger_optimizer=None):
    blob = {'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng': _rng_state(),
            'cfg': cfg,
            'cfg_fingerprint': _config_fingerprint(cfg)}
    if trigger is not None:
        blob['trigger'] = trigger.state_dict()
        blob['trigger_optimizer'] = trigger_optimizer.state_dict()
    torch.save(blob, path)
    print(f'[ckpt] saved -> {path}  (epoch {epoch})', flush=True)


def _load_checkpoint(path, model, optimizer, device, expected_cfg,
                     trigger=None, trigger_optimizer=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stored = ckpt.get('cfg_fingerprint')
    if stored is None:
        raise ValueError('checkpoint predates config fingerprinting')
    expected = _config_fingerprint(expected_cfg)
    if stored != expected:
        raise ValueError(f'config mismatch (stored={stored[:12]}, '
                         f'expected={expected[:12]})')
    # epochs is deliberately outside the fingerprint so a run can be extended,
    # but a checkpoint trained PAST the requested budget is a different model:
    # reusing it would report e.g. 200-epoch metrics for an epochs=100 request.
    done = ckpt['epoch'] + 1
    want = expected_cfg.get('epochs')
    if want is not None and done > want:
        raise ValueError(f'checkpoint has {done} epochs but this run asks for '
                         f'{want}; refusing to evaluate an over-trained model')
    if trigger is not None:
        if 'trigger' not in ckpt or 'trigger_optimizer' not in ckpt:
            raise ValueError('checkpoint is missing the learned-trigger state')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if trigger is not None:
        trigger.load_state_dict(ckpt['trigger'])
        trigger_optimizer.load_state_dict(ckpt['trigger_optimizer'])
    _restore_rng_state(ckpt.get('rng'))
    print(f'[ckpt] resumed from epoch {ckpt["epoch"]}', flush=True)
    return ckpt['epoch']

# ── Evaluation cache ─────────────────────────────────────────────────────────
# A finished cell should cost nothing on a re-run. Resuming only skips training,
# so without this the dose grid (6 doses x the test set) is recomputed for every
# already-done cell. Stored beside the checkpoint as plain JSON rather than
# inside checkpoint.pt, so a corrupt or stale cache never risks the weights.

_RESULT_SCHEMA = 8   # resolved-config/data-contract revision

_VICTIM_LOSS = 'mpjpe'
_ATTACK_SPECIFIC_VICTIM_KEYS = {
    'lambda_target', 'lambda_nontarget', 'loss_norm',
}


def _resolve_training_config(cfg):
    """Materialize the PiW3D paper recipe before hashing or training."""
    resolved = dict(cfg)
    resolved.pop('config_fingerprint', None)
    resolved.setdefault('experiment_name', 'one-person')
    resolved.setdefault('num_person', 1)
    resolved.setdefault('pretrained', False)
    resolved.setdefault('data_parallel', False)
    resolved.setdefault('seed', 42)
    resolved.setdefault('victim_loss', _VICTIM_LOSS)
    resolved.setdefault('trigger', 'micro_dropper')
    resolved.setdefault('poison_select', 'uniform')
    resolved.setdefault('dose_min', 0.2)
    resolved.setdefault('dose_max', 1.0)
    resolved.setdefault('dose_mode', 'linear')
    resolved.setdefault('dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    resolved.setdefault('tau_plaus', 0.20)
    resolved.setdefault('max_target_residual_ratio', 0.50)
    resolved.setdefault('max_nontarget_ratio', 0.25)
    resolved.setdefault('n_ant', 3)
    resolved.setdefault('n_sub', 30)
    resolved.setdefault('n_pkt', 20)
    resolved.setdefault('aoa_spread', 0.6)
    resolved.setdefault('top_k', 6)
    trigger_name = str(resolved['trigger']).lower().replace('-', '_')
    if trigger_name in ('tsba', 'tsba_adapted'):
        resolved.setdefault('tsba_eps', 0.1)
        resolved.setdefault('tsba_hidden', 32)
        resolved.setdefault('tsba_lr', 1e-3)
        resolved.setdefault('tsba_warmup_epochs', 10)
        resolved.setdefault('tsba_generator_steps', 4)
    resolved.setdefault('optimizer', 'adamw')
    resolved.setdefault('weight_decay', 0.01)
    if str(resolved['optimizer']).lower() != 'adamw':
        raise ValueError(
            'the PiW3D paper recipe requires optimizer=adamw; got '
            f"{resolved['optimizer']!r}")
    return resolved


def _validate_training_contract(cfg):
    """Reject victim-training options that exceed the paper threat model."""
    victim_loss = str(cfg.get('victim_loss', _VICTIM_LOSS)).lower()
    if victim_loss != _VICTIM_LOSS:
        raise ValueError(
            f"victim_loss must be '{_VICTIM_LOSS}' for paper experiments, "
            f'got {victim_loss!r}')
    stale = sorted(_ATTACK_SPECIFIC_VICTIM_KEYS.intersection(cfg))
    if stale:
        raise ValueError(
            'attack-specific victim-loss options are not allowed under the '
            f'paper threat model: {stale}. Remove them and retrain with MPJPE.')


def _result_path(ckpt_dir):
    # Deliberately NOT 'result.json'. This repo has no runner that writes that
    # name today, but the sibling ATKBackd runner does, and its public write
    # silently clobbered this cache's metadata — every re-run then reported a
    # schema mismatch and re-evaluated. Keeping the name reserved here keeps the
    # two repos interchangeable.
    return None if ckpt_dir is None else Path(ckpt_dir) / 'eval_cache.json'


def _load_cached_result(ckpt_dir, cfg):
    """Return the cached metrics for this exact config+budget, else None."""
    p = _result_path(ckpt_dir)
    if p is None or not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        print(f'[cache] unreadable ({e}); re-evaluating', flush=True)
        return None
    if blob.get('result_schema') != _RESULT_SCHEMA:
        print('[cache] metric schema changed; re-evaluating', flush=True)
        return None
    if blob.get('cfg_fingerprint') != _config_fingerprint(cfg):
        print('[cache] config changed; re-evaluating', flush=True)
        return None
    if blob.get('trained_epochs') != cfg.get('epochs'):
        print(f'[cache] epoch budget differs (cached '
              f'{blob.get("trained_epochs")} vs {cfg.get("epochs")}); '
              f're-evaluating', flush=True)
        return None
    res = blob.get('res')
    if not isinstance(res, dict):
        print('[cache] malformed payload; re-evaluating', flush=True)
        return None
    print(f'[cache] hit -> {p}  (skipping training and evaluation)', flush=True)
    return res


def _save_cached_result(ckpt_dir, cfg, res):
    p = _result_path(ckpt_dir)
    if p is None:
        return
    blob = {'result_schema': _RESULT_SCHEMA,
            'cfg_fingerprint': _config_fingerprint(cfg),
            'trained_epochs': cfg.get('epochs'),
            'cfg': cfg,
            'res': res}
    tmp = p.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(blob, indent=2, default=str), encoding='utf-8')
        os.replace(tmp, p)          # atomic: a killed process leaves no half file
        print(f'[cache] wrote -> {p}', flush=True)
    except OSError as e:
        print(f'[cache] could not write ({e}); continuing', flush=True)
        tmp.unlink(missing_ok=True)


def _write_run_metadata(ckpt_dir, cfg, poisoned_dataset):
    """Persist the resolved config and private poison plan for auditing."""
    if ckpt_dir is None:
        return
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = dict(cfg)
    (out / 'config.resolved.yaml').write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding='utf-8')
    manifest = poisoned_dataset.manifest()
    manifest['config_fingerprint'] = _config_fingerprint(cfg)
    (out / 'poison_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')


def build_trigger(cfg):
    name = cfg.get('trigger', 'micro_dropper').lower().replace('-', '_')
    if name in ('tsba', 'tsba_adapted'):
        return build_tsba_trigger(cfg)
    if name not in ('micro_dropper', 'microdropper', 'micro_doppler'):
        raise ValueError(
            f'unknown trigger {cfg.get("trigger")!r}; expected micro_dropper or tsba')
    t = MicroDopplerTrigger(aoa_spread=cfg.get('aoa_spread', 0.6), seed=cfg.get('seed', 0))
    vel, pos, movers = velocity_profiles_from_skeleton(cfg['action_npy'],
                                                       top_k=cfg.get('top_k', 6))
    t.build(vel, pos)
    return t


def _uses_deferred_trigger(trig):
    return bool(getattr(trig, 'requires_deferred_injection', False))


def _trigger_eps(cfg, trig):
    # TSBA's public implementation uses a 0.1 multiplicative bound.  Keep it
    # separate from micro-Doppler's eps=0.3 so selecting the baseline through a
    # runner cannot accidentally give TSBA three times its original budget.
    return float(cfg.get('tsba_eps', 0.1) if _uses_deferred_trigger(trig)
                 else cfg['eps'])


def _inject_deferred(trig, csi, dose, cfg):
    if not _uses_deferred_trigger(trig):
        return csi
    return trig.inject_tensor(csi, dose, eps=_trigger_eps(cfg, trig))


def _mpjpe_loss(pred, target):
    """Standard MPJPE for ordinary victim ERM.

    The API intentionally accepts no poison mask, dose, pivot, or target-joint
    identity, so attack metadata cannot affect the victim gradient.
    """
    if pred.shape != target.shape or pred.ndim < 3 or pred.shape[-1] != 3:
        raise ValueError(f'pred/target must have matching pose shapes, got '
                         f'{tuple(pred.shape)} and {tuple(target.shape)}')
    return torch.linalg.vector_norm(pred - target, dim=-1).mean()


def _victim_update(model, optimizer, csi, target):
    """One ordinary ERM step; attack metadata cannot enter this API."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(csi)
    loss = _mpjpe_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _generator_update(model, trig, optimizer, csi, target, dose, cfg):
    """One differentiable TSBA update with the victim frozen."""
    was_training = model.training
    requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    trig.train()
    optimizer.zero_grad()
    attacked = _inject_deferred(trig, csi, dose, cfg)
    pred, _ = model(attacked)
    loss = _mpjpe_loss(pred, target)
    loss.backward()
    optimizer.step()
    for p, flag in zip(model.parameters(), requires_grad):
        p.requires_grad_(flag)
    model.train(was_training)
    return float(loss.item())


@torch.no_grad()
def _predict(model, loader, device, trig=None, cfg=None):
    model.eval()
    preds, trues, targets, doses = [], [], [], []
    for b in loader:
        csi = b['csi'].to(device)
        if trig is not None and _uses_deferred_trigger(trig):
            trig.eval()
            csi = _inject_deferred(trig, csi, b['dose'].to(device), cfg)
        out, _ = model(csi)
        preds.append(out.cpu().numpy())
        trues.append(b['pose'].numpy())
        if 'target' in b: targets.append(b['target'].numpy())
        if 'dose' in b: doses.append(b['dose'].numpy())
    P = np.concatenate(preds)[:, 0]                       # (N,14,3) one person
    Tr = np.concatenate(trues)[:, 0]
    Tg = np.concatenate(targets)[:, 0] if targets else None
    Do = np.concatenate(doses) if doses else None
    return P, Tr, Tg, Do


def evaluate(model, base_test, trig, cfg, device):
    pivot = cfg['pivot']
    dl = lambda ds: DataLoader(ds, batch_size=cfg['batch_size'], collate_fn=collate)

    # clean
    clean_ds = PoisonedDataset(base_test, trig, mode='clean', pivot=pivot)
    Pc, Tc, _, _ = _predict(model, dl(clean_ds), device)
    res = {'clean_mpjpe': float(M.mpjpe(Pc, Tc).mean()),
           'clean_pampjpe': float(M.pa_mpjpe(Pc, Tc).mean()),
           'clean_pck@0.5': M.pck(Pc, Tc, 0.5)}

    # dose-response: per-dose displacement, t-MPJPE (vs attacker target), localization, plausibility
    grid = list(map(float, cfg.get(
        'dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])))
    if len(grid) < 2 or any(d < 0.0 or d > 1.0 for d in grid):
        raise ValueError(f'dose_grid must contain at least two doses in [0,1], got {grid}')
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f'dose_grid must be strictly increasing, got {grid}')
    ref_matches = [i for i, d in enumerate(grid) if np.isclose(d, 1.0)]
    if len(ref_matches) != 1:
        raise ValueError('dose_grid must contain dose=1.0 exactly once for ASR@ref')
    ref_idx = ref_matches[0]
    disp, tmp, tpa, loc, plaus, asr_curve = [], [], [], [], [], []
    for d in grid:
        tds = PoisonedDataset(base_test, trig, mode='trigger@dose',
                              pivot=pivot, fixed_dose=d, eps=cfg['eps'],
                              theta_max_deg=cfg['theta_max_deg'],
                              dose_mode=cfg['dose_mode'])
        Pd, Td, Tg, _ = _predict(model, dl(tds), device, trig=trig, cfg=cfg)
        disp.append(M.subchain_displacement(Pd, Pc, pivot))
        tmp.append(float(M.target_mpjpe(Pd, Tg, pivot).mean()))
        tpa.append(float(M.target_pa_mpjpe(Pd, Tg, pivot).mean()))
        loc.append(M.nontarget_preservation(Pd, Pc, pivot))
        plaus.append(M.plausibility_error(Pd, Td, pivot))
        asr_curve.append(M.attack_metrics(
            Pd, Tg, Pc, Tc, pivot,
            max_target_residual_ratio=cfg.get('max_target_residual_ratio', 0.50),
            max_nontarget_ratio=cfg.get('max_nontarget_ratio', 0.25),
            tau_plaus=cfg.get('tau_plaus', 0.20)))
    tfloor, nfloor = M.clean_floor(Pc, Tc, pivot)
    res['dose_grid'] = grid
    res['displacement'] = list(map(float, disp))
    res['tmpjpe'] = list(map(float, tmp))
    res['tpampjpe'] = list(map(float, tpa))   # Procrustes-aligned T-MPJPE
    res['nontarget_mpjpe'] = list(map(float, loc))
    res['plausibility'] = list(map(float, plaus))
    res['clean_target_floor'] = tfloor                        # model noise floor on the limb
    res['dose_response'] = M.dose_response_analysis(grid, disp)
    res['schedule_shape'] = M.schedule_shape_analysis(
        grid, disp, cfg['dose_mode'], reference_dose=grid[ref_idx])
    res['asr_curve'] = asr_curve
    res['reference_dose'] = grid[ref_idx]
    res['asr@ref'] = asr_curve[ref_idx]
    return res


def train(cfg, ckpt_dir=None):
    """Train one poisoned model and evaluate it.

    ckpt_dir: when given, this run checkpoints there and resumes from it if the
    stored config fingerprint matches, so an interrupted sweep loses at most one
    cell's partial progress rather than the whole grid.
    """
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)

    # A finished cell returns immediately — before touching the dataset, the
    # model or the trigger — so re-running a sweep really costs nothing for the
    # cells already done. Returns (None, res): callers that only read metrics
    # (sweep.py) are fine; anyone needing the model must clear the cache.
    cached = _load_cached_result(ckpt_dir, cfg)
    if cached is not None:
        return None, cached

    device = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    base_train = PersonInWiFi3D('training', cfg['dataset_root'], cfg['experiment_name'])
    base_test = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    trig = build_trigger(cfg)
    learned_trigger = _uses_deferred_trigger(trig)
    if learned_trigger:
        trig = trig.to(device)

    pois = PoisonedDataset(base_train, trig, mode='train', rho=cfg['rho'],
                           dose_min=cfg['dose_min'], dose_max=cfg['dose_max'],
                           eps=cfg['eps'], pivot=cfg['pivot'],
                           theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'],
                           seed=cfg.get('seed', 0), select=cfg.get('poison_select', 'uniform'))
    _write_run_metadata(ckpt_dir, cfg, pois)
    loader = DataLoader(pois, batch_size=cfg['batch_size'], shuffle=True,
                        collate_fn=collate, drop_last=False)

    model = build_model(cfg['model'], subcarrier_num=180,
                        pretrained=cfg.get('pretrained', False)).to(device)
    if (cfg.get('data_parallel') and str(device).startswith('cuda')
            and torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    print(f'[train] optimizer=adamw lr={cfg["lr"]} '
          f'victim_loss={_VICTIM_LOSS} (ordinary ERM)', flush=True)
    trig_opt = (torch.optim.Adam(trig.parameters(),
                                lr=float(cfg.get('tsba_lr', 1e-3)))
                if learned_trigger else None)

    # ── Resume, if this cell already has a matching checkpoint ──────────
    ckpt_file = _ckpt_path(cfg, ckpt_dir)
    start_epoch = 0
    if ckpt_file is not None and ckpt_file.exists():
        try:
            start_epoch = _load_checkpoint(
                ckpt_file, model, opt, device, cfg,
                trigger=trig if learned_trigger else None,
                trigger_optimizer=trig_opt) + 1
        except ValueError as e:
            print(f'[ckpt] not reusable ({e}); starting fresh', flush=True)
            start_epoch = 0

    total_epochs = cfg['epochs']
    ckpt_every = cfg.get('ckpt_every', 10)
    if start_epoch >= total_epochs:
        print(f'[ckpt] already trained {start_epoch}/{total_epochs} epochs; '
              f'skipping to evaluation', flush=True)

    for epoch in range(start_epoch, total_epochs):
        model.train(); losses = []; generator_losses = []
        for b in loader:
            csi, pose = b['csi'].to(device), b['pose'].to(device)
            if learned_trigger:
                poisoned = b['poisoned'].to(device).bool()
                dose = b['dose'].to(device)
                warmup = epoch < int(cfg.get('tsba_warmup_epochs', 10))

                if not warmup and bool(poisoned.any()):
                    steps = int(cfg.get('tsba_generator_steps', 4))
                    for _ in range(steps):
                        generator_losses.append(_generator_update(
                            model, trig, trig_opt, csi[poisoned], pose[poisoned],
                            dose[poisoned], cfg))

                # The victim sees the same poisoned subset/target labels as the
                # micro-Doppler run.  During clean warm-up, every sample keeps
                # its original CSI and pose.
                if warmup:
                    train_csi = csi
                    train_pose = b['clean_pose'].to(device)
                else:
                    train_csi = csi.clone()
                    if bool(poisoned.any()):
                        trig.eval()
                        with torch.no_grad():
                            train_csi[poisoned] = _inject_deferred(
                                trig, csi[poisoned], dose[poisoned], cfg)
                    train_pose = pose
            else:
                train_csi, train_pose = csi, pose

            losses.append(_victim_update(model, opt, train_csi, train_pose))
        msg = f'epoch {epoch}: train_loss={np.mean(losses):.4f}'
        if generator_losses:
            msg += f' tsba_loss={np.mean(generator_losses):.4f}'
        print(msg)
        if ckpt_file is not None and (
                (epoch + 1) % ckpt_every == 0 or epoch == total_epochs - 1):
            _save_checkpoint(
                ckpt_file, model, opt, epoch, cfg,
                trigger=trig if learned_trigger else None,
                trigger_optimizer=trig_opt)

    res = evaluate(model, base_test, trig, cfg, device)
    res['n_poison'] = int(pois.n_poison)
    res['n_total'] = int(pois.n_total)
    res['dataset'] = 'person-in-wifi-3d'
    res['pivot'] = int(cfg['pivot'])
    res['theta_max_deg'] = float(cfg['theta_max_deg'])
    res['dose_mode'] = cfg['dose_mode']
    res['rho'] = float(cfg['rho'])
    res['eps'] = float(cfg['eps'])
    res['poison_select'] = cfg.get('poison_select', 'uniform')
    res['seed'] = int(cfg['seed'])
    res['config_fingerprint'] = _config_fingerprint(cfg)
    res['poison_plan_sha256'] = pois.manifest()['poison_plan_sha256']
    res['victim_loss'] = _VICTIM_LOSS
    res['training_contract'] = 'ordinary_erm'
    res['attacker_access'] = ('white_box_training_control'
                              if learned_trigger else 'data_only')
    if learned_trigger:
        res['trigger'] = 'tsba_adapted'
        res['trigger_eps'] = _trigger_eps(cfg, trig)
        res['tsba_generator_parameters'] = sum(p.numel() for p in trig.parameters())
        # Keep the trained generator available to callers that render examples.
        object.__setattr__(model, '_trained_trigger', trig)
    _save_cached_result(ckpt_dir, cfg, res)
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--ckpt-dir', default=None,
                    help='Directory to save/load victim and learned-trigger state')
    a = ap.parse_args()
    with open(a.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _, res = train(cfg, ckpt_dir=a.ckpt_dir)
    print('\n==== RESULTS ===='); import json; print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
