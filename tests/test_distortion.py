"""Regression tests for eval/distortion.py.

These exist because the first version of this tool measured TSBA with a freshly
initialised generator and reported the resulting ratio as if it characterised
the trained attack. It does not: a learned trigger's perturbation is entirely a
property of its trained weights. Tests 4-7 are the guard against that.

Run:  python tests/test_distortion.py     (no pytest needed)
      pytest tests/test_distortion.py     (also works)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.distortion import (                                    # noqa: E402
    distortion_stats, _aggregate, load_trigger_for_measurement)
from train_backdoor import (build_trigger, _config_fingerprint,
                            _resolve_training_config)             # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name):
    """Load a shipped config without consulting machine-specific assets."""
    for rel in (f'configs/{name}', f'configs/mmfi/{name}'):
        p = os.path.join(_HERE, rel)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as stream:
                cfg = yaml.safe_load(stream)
            break
    else:
        raise FileNotFoundError(name)
    return cfg


def _tsba_cfg():
    for n in ('piw_p3_th90_tsba_unif.yaml', 'tsba_colab.yaml',
              'attack_tsba_bend.yaml'):
        try:
            return _cfg(n)
        except FileNotFoundError:
            continue
    raise FileNotFoundError('no TSBA config found')


def _fixed_cfg(action_npy):
    for n in ('piw_p3_th90_uniform.yaml', 'sweep_colab.yaml',
              'attack_bend.yaml'):
        try:
            cfg = _cfg(n)
            cfg['action_npy'] = str(action_npy)
            return cfg
        except FileNotFoundError:
            continue
    raise FileNotFoundError('no micro-Doppler config found')


@pytest.fixture
def action_npy():
    """Small NTU-shaped moving skeleton; no workstation/Colab asset needed."""
    # Use the repository's writable tests directory rather than pytest's global
    # temp root, which is permission-restricted on some Windows runners.
    with tempfile.TemporaryDirectory(
            prefix='_action_fixture_', dir=os.path.dirname(__file__)) as tmp:
        n_frames, n_joints = 8, 8
        action = np.zeros((1, 3, n_frames, n_joints, 1), dtype=np.float32)
        time = np.arange(n_frames, dtype=np.float32)
        for joint in range(n_joints):
            action[0, 0, :, joint, 0] = (
                0.1 * joint + 0.01 * (joint + 1) * time)
            action[0, 1, :, joint, 0] = (
                0.2 + 0.02 * joint + 0.005 * time)
            action[0, 2, :, joint, 0] = 1.0 + 0.03 * joint
        path = Path(tmp) / 'action_fixture.npy'
        np.save(path, action)
        yield path


# ── 1. identical inputs give infinite SNR, not a large finite number ─────────
def test_zero_error_is_infinite_snr():
    x = np.random.RandomState(0).rand(3, 20, 10)
    s = distortion_stats(x, x)
    assert s['snr_db'] == float('inf'), f'expected inf, got {s["snr_db"]}'
    assert s['relative_l2'] == 0.0
    assert s['max_abs'] == 0.0


# ── 2. a known gain has the textbook relative_l2 and SNR ────────────────────
def test_known_gain_matches_theory():
    x = np.random.RandomState(0).rand(3, 20, 10)
    s = distortion_stats(x, x * 1.1)
    assert abs(s['relative_l2'] - 0.1) < 1e-9, s['relative_l2']
    assert abs(s['snr_db'] - 20.0) < 1e-6, s['snr_db']


# ── 3. dataset stats pool energy; max is a true max ─────────────────────────
def test_aggregate_pools_energy_not_ratios():
    rs = np.random.RandomState(0)
    big, small = rs.rand(100) * 10.0, rs.rand(100) * 0.1
    r1 = distortion_stats(big, big + 1.0)
    r2 = distortion_stats(small, small + 1.0)      # same error, tiny signal
    agg = _aggregate([r1, r2])

    mean_ratio = agg['mean_per_sample_relative_l2']
    assert agg['relative_l2'] < mean_ratio, (
        'pooled relative_l2 must not equal the mean of per-sample ratios; '
        f'pooled={agg["relative_l2"]} mean={mean_ratio}')
    assert abs(agg['max_abs'] - max(r1['max_abs'], r2['max_abs'])) < 1e-12, \
        'max_abs must be a true dataset maximum, not a mean of per-sample maxima'


# ── 4. a learned trigger is refused without its trained weights ─────────────
def test_learned_trigger_requires_checkpoint():
    cfg = _tsba_cfg()
    try:
        load_trigger_for_measurement(cfg)
    except SystemExit:
        return
    raise AssertionError('TSBA must not be measurable without --checkpoint')


# ── 5. the escape hatch works, and is explicit ──────────────────────────────
def test_allow_untrained_escape_hatch():
    cfg = _tsba_cfg()
    trig, epoch = load_trigger_for_measurement(cfg, allow_untrained=True)
    assert trig is not None and epoch is None


# ── 6. a fixed trigger needs no checkpoint ──────────────────────────────────
def test_fixed_trigger_needs_no_checkpoint(action_npy):
    trig, epoch = load_trigger_for_measurement(_fixed_cfg(action_npy))
    assert trig is not None and epoch is None


# ── 7. checkpoint loading: right weights, wrong config refused ──────────────
def test_checkpoint_load_and_fingerprint_guard():
    cfg = _tsba_cfg()
    resolved = _resolve_training_config(cfg)
    tmp = tempfile.mkdtemp()
    try:
        trained = build_trigger(resolved)
        with torch.no_grad():                       # pretend it was trained
            for p in trained.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ck = os.path.join(tmp, 'checkpoint.pt')
        torch.save({'epoch': 49, 'model': {}, 'optimizer': {}, 'cfg': resolved,
                    'cfg_fingerprint': _config_fingerprint(resolved),
                    'trigger': trained.state_dict(),
                    'trigger_optimizer': {}}, ck)

        got, epoch = load_trigger_for_measurement(cfg, checkpoint=ck)
        assert epoch == 49
        for a, b in zip(trained.state_dict().values(), got.state_dict().values()):
            assert torch.allclose(a, b), 'trained generator weights were not loaded'

        try:                                        # different config must refuse
            load_trigger_for_measurement(dict(cfg, pivot=6), checkpoint=ck)
        except SystemExit:
            pass
        else:
            raise AssertionError('a config mismatch must be refused')

        md = os.path.join(tmp, 'md.pt')             # checkpoint with no trigger
        torch.save({'epoch': 10, 'model': {}, 'optimizer': {}, 'cfg': resolved,
                    'cfg_fingerprint': _config_fingerprint(resolved)}, md)
        try:
            load_trigger_for_measurement(cfg, checkpoint=md)
        except SystemExit:
            pass
        else:
            raise AssertionError('a checkpoint without trigger state must be refused')
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    # Delegate to pytest so fixtures (including the temporary action skeleton)
    # behave identically whether this file or the complete suite is invoked.
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
