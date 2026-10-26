"""Regression tests for checkpoint resume and the evaluation cache.

Run:  python tests/test_checkpoint.py        (no pytest needed)
      pytest tests/test_checkpoint.py        (also works)

These exist because an ad-hoc check caught a real defect: resume restored the
weights bit-perfectly but NOT the RNG state, so a resumed run replayed epoch 0's
batch order and drifted from an uninterrupted run (~2.4e-3 parameter difference).
Test 1 is the guard against that regression.
"""
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_backdoor import (                                    # noqa: E402
    _config_fingerprint, _ckpt_path, _save_checkpoint, _load_checkpoint,
    _load_cached_result, _save_cached_result, _result_path,
    _RESULT_SCHEMA,
)
from attack.tsba import TSBATrigger                              # noqa: E402

BASE_CFG = dict(model='hpeli', pivot=7, theta_max_deg=40.0, rho=0.4, eps=0.3,
                dose_mode='linear', poison_select='diverse', seed=42, lr=0.01,
                epochs=4)


# ── helpers ──────────────────────────────────────────────────────────────────
class _Toy(Dataset):
    def __init__(self, n=64):
        self.x = torch.randn(n, 8)
        self.y = torch.randn(n, 3)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _seed_all(sd=42):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)


def _model():
    _seed_all()
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                               torch.nn.Linear(16, 3))


def _save(path, model, opt, epoch, cfg):
    """_save_checkpoint has a best_loss arg in one repo but not the other."""
    import inspect
    params = inspect.signature(_save_checkpoint).parameters
    if 'best_loss' in params:
        _save_checkpoint(path, model, opt, epoch, 0.0, cfg)
    else:
        _save_checkpoint(path, model, opt, epoch, cfg)


def _resumed_epoch(ret):
    """_load_checkpoint returns epoch, or (epoch, best_loss)."""
    return ret[0] if isinstance(ret, tuple) else ret


def _train_epochs(model, opt, ds, first, last, ckpt=None, cfg=None):
    # shuffle=True with no explicit generator: uses the global torch RNG, which
    # is exactly what train() does, and what the RNG checkpoint must cover.
    for ep in range(first, last):
        model.train()
        for xb, yb in DataLoader(ds, batch_size=8, shuffle=True):
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ckpt is not None:
            _save(ckpt, model, opt, ep, cfg)


def _max_param_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(),
                                                          b.parameters()))


# ── 1. uninterrupted == resumed ──────────────────────────────────────────────
def test_resume_matches_uninterrupted():
    cfg = dict(BASE_CFG, epochs=4)
    tmp = tempfile.mkdtemp()
    try:
        ckpt = _ckpt_path(cfg, tmp)
        _seed_all()
        ds = _Toy()

        _seed_all()
        m_full = _model()
        o_full = torch.optim.AdamW(m_full.parameters(), lr=cfg['lr'])
        _seed_all()
        _train_epochs(m_full, o_full, ds, 0, 4)

        _seed_all()
        m_part = _model()
        o_part = torch.optim.AdamW(m_part.parameters(), lr=cfg['lr'])
        _seed_all()
        _train_epochs(m_part, o_part, ds, 0, 2, ckpt=ckpt, cfg=cfg)

        m_res = _model()
        o_res = torch.optim.AdamW(m_res.parameters(), lr=cfg['lr'])
        ep = _resumed_epoch(_load_checkpoint(ckpt, m_res, o_res, 'cpu', cfg))
        _train_epochs(m_res, o_res, ds, ep + 1, 4)

        diff = _max_param_diff(m_full, m_res)
        assert diff == 0.0, f'resume diverged: max param diff {diff:.3e}'
    finally:
        shutil.rmtree(tmp)


# ── 2. changing a result-affecting option is refused ─────────────────────────
def test_fingerprint_rejects_changed_config():
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG)
        ckpt = _ckpt_path(cfg, tmp)
        m = _model()
        o = torch.optim.AdamW(m.parameters(), lr=cfg['lr'])
        _save(ckpt, m, o, 1, cfg)

        for key, val in [('pivot', 6), ('theta_max_deg', 60.0), ('rho', 0.1),
                         ('eps', 0.5), ('seed', 0),
                         ('poison_select', 'uniform'), ('dose_mode', 'sqrt'),
                         ('lambda_target', 2.0),
                         ('lambda_nontarget', 0.5)]:
            m2 = _model()
            o2 = torch.optim.AdamW(m2.parameters(), lr=cfg['lr'])
            try:
                _load_checkpoint(ckpt, m2, o2, 'cpu', dict(cfg, **{key: val}))
            except ValueError:
                continue
            raise AssertionError(f'changing {key} should have been refused')
    finally:
        shutil.rmtree(tmp)


# ── 3. epoch budget: extending is fine, shrinking is not ─────────────────────
def test_epoch_budget_guard():
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG, epochs=200)
        ckpt = _ckpt_path(cfg, tmp)
        m = _model()
        o = torch.optim.AdamW(m.parameters(), lr=cfg['lr'])
        _save(ckpt, m, o, 199, cfg)                    # 200 epochs done

        for want in (200, 500):                        # same or extended: OK
            m2 = _model()
            o2 = torch.optim.AdamW(m2.parameters(), lr=cfg['lr'])
            _load_checkpoint(ckpt, m2, o2, 'cpu', dict(cfg, epochs=want))

        for want in (100, 50):                         # shrunk: must refuse
            m2 = _model()
            o2 = torch.optim.AdamW(m2.parameters(), lr=cfg['lr'])
            try:
                _load_checkpoint(ckpt, m2, o2, 'cpu', dict(cfg, epochs=want))
            except ValueError:
                continue
            raise AssertionError(
                f'epochs={want} against a 200-epoch checkpoint should be refused')
    finally:
        shutil.rmtree(tmp)


# ── 4. cache hit on matching fingerprint + epoch ─────────────────────────────
def test_result_cache_hit():
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG, epochs=200)
        res = {'clean_mpjpe': 0.134, 'asr@ref': {'asr': 0.5}}
        _save_cached_result(tmp, cfg, res)
        assert _result_path(tmp).exists(), 'cache file was not written'
        got = _load_cached_result(tmp, cfg)
        assert got == res, f'cache round-trip changed the payload: {got}'
    finally:
        shutil.rmtree(tmp)


# ── 5. cache misses on schema / config / epoch change, and on corruption ─────
def test_result_cache_misses():
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG, epochs=200)
        _save_cached_result(tmp, cfg, {'clean_mpjpe': 0.134})

        assert _load_cached_result(tmp, dict(cfg, pivot=6)) is None, \
            'changed config must miss'
        assert _load_cached_result(tmp, dict(cfg, epochs=100)) is None, \
            'changed epoch budget must miss'

        p = _result_path(tmp)
        blob = json.loads(p.read_text())
        blob['result_schema'] = _RESULT_SCHEMA + 1
        p.write_text(json.dumps(blob))
        assert _load_cached_result(tmp, cfg) is None, \
            'bumped metric schema must miss'

        p.write_text('{not json')
        assert _load_cached_result(tmp, cfg) is None, \
            'corrupt cache must miss, not raise'
    finally:
        shutil.rmtree(tmp)


# ── 6. a cell with no ckpt_dir must not crash or cache ───────────────────────
def test_no_ckpt_dir_is_a_noop():
    cfg = dict(BASE_CFG)
    assert _result_path(None) is None
    assert _load_cached_result(None, cfg) is None
    _save_cached_result(None, cfg, {'x': 1})           # must not raise


# ── 7. the public result.json must not collide with the eval cache ───────────
def test_cache_survives_public_result_json():
    """run_experiments.py writes cell_dir/result.json after train() returns.

    The eval cache lives in the same directory, so if it were also named
    result.json the public write would clobber its metadata and every re-run
    would report a schema mismatch and re-evaluate.
    """
    import pathlib
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG, epochs=200)
        res = {'clean_mpjpe': 0.134}
        _save_cached_result(tmp, cfg, res)

        cache = _result_path(tmp)
        assert cache.name != 'result.json', (
            'eval cache must not be named result.json — run_experiments.py '
            'writes the public per-run result under that name')

        # simulate the runner's public write, exactly as run_experiments does
        (pathlib.Path(tmp) / 'result.json').write_text(json.dumps(res, indent=2))

        assert _load_cached_result(tmp, cfg) == res, (
            'writing the public result.json destroyed the evaluation cache')
    finally:
        shutil.rmtree(tmp)


def test_learned_trigger_checkpoint_round_trip():
    """A resumed TSBA cell must restore generator weights and Adam state."""
    import inspect
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(BASE_CFG, trigger='tsba', tsba_hidden=8, tsba_eps=0.1)
        ckpt = _ckpt_path(cfg, tmp)
        model = _model()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        trigger = TSBATrigger(hidden=8)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=1e-3)
        trigger_opt.zero_grad()
        trigger(torch.rand(2, 3, 12, 5)).square().mean().backward()
        trigger_opt.step()

        params = inspect.signature(_save_checkpoint).parameters
        if 'best_loss' in params:
            _save_checkpoint(ckpt, model, opt, 1, 0.0, cfg,
                             trigger=trigger, trigger_optimizer=trigger_opt)
        else:
            _save_checkpoint(ckpt, model, opt, 1, cfg,
                             trigger=trigger, trigger_optimizer=trigger_opt)

        model2 = _model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=cfg['lr'])
        trigger2 = TSBATrigger(hidden=8)
        trigger_opt2 = torch.optim.Adam(trigger2.parameters(), lr=1e-3)
        _load_checkpoint(ckpt, model2, opt2, 'cpu', cfg,
                         trigger=trigger2, trigger_optimizer=trigger_opt2)
        for key, value in trigger.state_dict().items():
            torch.testing.assert_close(value, trigger2.state_dict()[key],
                                       rtol=0.0, atol=0.0)
        assert trigger_opt2.state_dict()['state'], 'generator Adam state was lost'
    finally:
        shutil.rmtree(tmp)

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:                          # noqa: BLE001
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
