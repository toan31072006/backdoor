"""Measure how much a trigger actually perturbs the tensor the victim sees.

The `eps` of two triggers is not comparable: micro-Doppler's eps scales an
RMS-normalised complex field and is injected BEFORE preprocessing (so it still
passes through per-sample min-max on Person-in-WiFi-3D, or a clip on MMFi),
while TSBA's eps is a per-element bound on tanh(G(x)) applied AFTER
normalisation. Reporting "eps=0.3 vs 0.1" therefore says nothing about relative
strength: the only meaningful comparison is measured on the model input.

TSBA's perturbation depends entirely on its LEARNED generator, so measuring it
with a freshly initialised generator says nothing about the trained attack —
the numbers would be an artefact of the random init. This tool therefore
refuses to measure a learned trigger without --checkpoint unless the caller
explicitly passes --allow-untrained, and any figure produced that way must not
be reported.

    # micro-Doppler is a fixed trigger: no checkpoint needed
    python -m eval.distortion --config configs/sweep_colab.yaml

    # TSBA must load the generator trained for THIS cell
    python -m eval.distortion --config configs/tsba_colab.yaml \\
        --checkpoint sweep_out/checkpoints/tsba_theta_40_rho_0p4/checkpoint.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def distortion_stats(x0, x1):
    """Per-sample distortion of x1 relative to x0, as the victim receives them.

    relative_l2 is the headline number: it is scale-free, so it compares
    triggers whose eps live in different units. SNR is the same quantity in dB.
    Returns raw energies too, so a caller can aggregate over a dataset properly
    (summing energies) instead of averaging per-sample ratios.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x1 - x0
    sig_energy = float((x0 ** 2).sum())
    err_energy = float((d ** 2).sum())
    n = d.size
    return {
        'relative_l2': float(np.sqrt(err_energy / (sig_energy + 1e-300))),
        'rms': float(np.sqrt(err_energy / n)),
        'mae': float(np.abs(d).mean()),
        'snr_db': (float('inf') if err_energy == 0.0
                   else float(10 * np.log10(sig_energy / err_energy))),
        'max_abs': float(np.abs(d).max()),
        # for dataset-level aggregation
        '_sig_energy': sig_energy,
        '_err_energy': err_energy,
        '_abs_sum': float(np.abs(d).sum()),
        '_n': int(n),
    }


def _aggregate(rows):
    """Dataset-level statistics: pool energies, do not average ratios.

    Averaging per-sample relative_l2 or SNR weights every sample equally
    regardless of its energy, which is not the dataset SNR. max_abs must be a
    true maximum over the dataset, not a mean of per-sample maxima.
    """
    sig = sum(r['_sig_energy'] for r in rows)
    err = sum(r['_err_energy'] for r in rows)
    n = sum(r['_n'] for r in rows)
    return {
        'relative_l2': float(np.sqrt(err / (sig + 1e-300))),
        'rms': float(np.sqrt(err / n)),
        'mae': float(sum(r['_abs_sum'] for r in rows) / n),
        'snr_db': float('inf') if err == 0.0 else float(10 * np.log10(sig / err)),
        'max_abs': max(r['max_abs'] for r in rows),
        # per-sample means, kept for reference — NOT the dataset figure
        'mean_per_sample_relative_l2': float(np.mean([r['relative_l2'] for r in rows])),
    }


def load_trigger_for_measurement(cfg, checkpoint=None, allow_untrained=False,
                                 device='cpu'):
    """Build the trigger, loading learned weights when the trigger has any.

    A fixed trigger (micro-Doppler) is fully determined by the config. A learned
    trigger (TSBA) is not: without its trained generator the measurement is
    meaningless, so it is refused unless explicitly overridden.
    """
    from train_backdoor import (build_trigger, _uses_deferred_trigger,
                                _config_fingerprint, _resolve_training_config,
                                _validate_training_contract)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    trig = build_trigger(cfg)
    learned = _uses_deferred_trigger(trig)

    if not learned:
        if checkpoint:
            print(f'[distortion] {cfg.get("trigger", "micro_dropper")} is a fixed '
                  f'trigger; ignoring --checkpoint')
        return trig, None

    if checkpoint is None:
        if not allow_untrained:
            raise SystemExit(
                f'Refusing to measure learned trigger '
                f'"{cfg.get("trigger")}" without --checkpoint: a freshly '
                f'initialised generator produces numbers that reflect the random '
                f'init, not the trained attack. Pass the cell checkpoint, or '
                f'--allow-untrained if you understand the result is not reportable.')
        print('[distortion] WARNING: measuring an UNTRAINED generator — '
              'these numbers are not reportable', flush=True)
        trig.eval()
        return trig, None

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    stored = ckpt.get('cfg_fingerprint')
    expected = _config_fingerprint(cfg)
    if stored is None:
        raise SystemExit(f'{checkpoint} has no cfg_fingerprint; it predates '
                         f'config isolation and cannot be trusted here')
    if stored != expected:
        raise SystemExit(
            f'checkpoint was trained under a different config '
            f'(stored={stored[:12]}, expected={expected[:12]}). Measuring it '
            f'against this config would mislabel the result.')
    if 'trigger' not in ckpt:
        raise SystemExit(f'{checkpoint} stores no trigger state — it is probably '
                         f'from a fixed-trigger run')
    trig.load_state_dict(ckpt['trigger'])
    trig.eval()
    print(f'[distortion] loaded trained generator from {checkpoint} '
          f'(epoch {ckpt["epoch"]})', flush=True)
    return trig, ckpt['epoch']


def measure(cfg, n=64, dose=1.0, split='validation', trig=None):
    """Dataset-level distortion over n real samples."""
    from train_backdoor import (_uses_deferred_trigger, _trigger_eps,
                                _resolve_training_config,
                                _validate_training_contract)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    try:                                        # ATKBackd exposes a loader
        from train_backdoor import _load_dataset, _get_dataset_name
        ds = _load_dataset(cfg, split if _get_dataset_name(cfg) != 'mmfi' else 'test')
    except ImportError:                         # wbackdoor: Person-in-WiFi-3D only
        from data_utils.feeder import PersonInWiFi3D
        ds = PersonInWiFi3D(split, cfg['dataset_root'], cfg['experiment_name'])

    if trig is None:
        trig, _ = load_trigger_for_measurement(cfg)
    deferred = _uses_deferred_trigger(trig)
    eps = _trigger_eps(cfg, trig)

    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    rows = []
    for i in idx:
        raw = ds.load_raw(ds.items[i]['csi'])
        clean = ds.normalize(raw)
        if deferred:
            # Injected after normalisation, straight onto the model input.
            with torch.no_grad():
                t = torch.from_numpy(clean)[None].float()
                hit = trig.inject_tensor(t, dose, eps=eps)[0].numpy()
        else:
            # Injected into raw CSI, then re-normalised like any other sample.
            hit = ds.normalize(trig.inject(raw, dose, eps=eps))
        rows.append(distortion_stats(clean, hit))

    out = _aggregate(rows)
    out.update(n_samples=len(rows), dose=float(dose), eps=float(eps),
               trigger=cfg.get('trigger', 'micro_dropper'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None,
                    help='cell checkpoint.pt — required for learned triggers (TSBA)')
    ap.add_argument('--allow-untrained', action='store_true',
                    help='measure a learned trigger without its trained weights '
                         '(diagnostic only; the result is NOT reportable)')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--dose', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    a = ap.parse_args()

    with open(a.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    from train_backdoor import (_resolve_training_config,
                                _validate_training_contract)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    trig, epoch = load_trigger_for_measurement(
        cfg, checkpoint=a.checkpoint, allow_untrained=a.allow_untrained)

    print(f"\ntrigger={cfg.get('trigger', 'micro_dropper')}  config={a.config}"
          + (f'  generator@epoch{epoch}' if epoch is not None else ''))
    print(f'{"dose":>6} {"relL2":>9} {"RMS":>9} {"MAE":>9} {"SNR dB":>9} {"max|d|":>9}')
    print('-' * 55)
    for d in a.dose:
        s = measure(cfg, n=a.n, dose=d, split=a.split, trig=trig)
        snr = '   inf' if s['snr_db'] == float('inf') else f'{s["snr_db"]:9.1f}'
        print(f'{d:6.2f} {s["relative_l2"]:9.4f} {s["rms"]:9.4f} {s["mae"]:9.4f} '
              f'{snr:>9} {s["max_abs"]:9.4f}')
    print(f'\n(eps={s["eps"]}, {s["n_samples"]} samples from "{a.split}"; '
          f'relL2/SNR are dataset-level (pooled energy), max|d| is a true max)')
    if a.allow_untrained and cfg.get('trigger') not in (None, 'micro_dropper'):
        print('WARNING: generator was untrained — do not report these numbers')


if __name__ == '__main__':
    main()
