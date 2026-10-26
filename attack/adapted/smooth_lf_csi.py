"""Fixed smooth low-frequency CSI control inspired by LF.

This is an independent MM-Fi amplitude/pose adaptation, *not* a reproduction
of LF's optimized image trigger. BackdoorBench LF loads a precomputed pattern;
the original LF authors generate that pattern with a pretrained clean image
classifier. Here the key is a seeded, fixed, low-frequency DCT pattern and
requires no surrogate model, labels, or test data. The additive operator is
the portion of LF's mechanism retained by this control.

Sources (attribution only; no source code copied):
  https://github.com/ethan-yi-zeng/frequency-backdoor
  https://github.com/SCLBD/BackdoorBench/blob/main/attack/lf.py

CSI shape is (antenna, subcarrier, packet), normally (3, 114, 10). The antenna
axis is a channel axis. This operator does not invent a phase measurement.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

from attack.adapted.amplitude_base import _AmplitudeTrigger, _positive_integer


class SmoothLFCSITrigger(_AmplitudeTrigger):
    """A fixed seeded DCT-low-pass additive trigger with an L-infinity cap."""

    baseline_name = 'Smooth-LF-CSI-control'
    dose_semantics = 'add clip(dose * eps * fixed_pattern, -dose*eps, dose*eps)'
    source_relation = 'LF-inspired smooth control; no clean-surrogate trigger optimization'

    def __init__(self, *, low_subcarriers=3, low_packets=2, **kwargs):
        super().__init__(**kwargs)
        self.low_subcarriers = _positive_integer(low_subcarriers, 'low_subcarriers')
        self.low_packets = _positive_integer(low_packets, 'low_packets')
        if self.low_subcarriers > self.n_sub or self.low_packets > self.n_pkt:
            raise ValueError('Low-frequency DCT window exceeds the CSI grid')
        if self.low_subcarriers * self.low_packets < 2:
            raise ValueError('At least one non-DC low-frequency coefficient is required')

        rng = np.random.default_rng(self.seed)
        coefficients = np.zeros(self.shape, dtype=np.float64)
        coefficients[:, :self.low_subcarriers, :self.low_packets] = rng.standard_normal(
            (self.n_ant, self.low_subcarriers, self.low_packets))
        coefficients[:, 0, 0] = 0.0  # no image-style brightness/CSI DC key
        pattern = idctn(coefficients, type=2, norm='ortho', axes=(-2, -1))
        peak = float(np.max(np.abs(pattern)))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError('Degenerate low-frequency key')
        self.pattern = (pattern / peak).astype(np.float64)

    def inject(self, csi, dose, eps=0.185):
        value, scale = self._input(csi, dose, eps)
        if dose > 1:
            raise ValueError('Dose must be in [0,1]')
        if scale == 0:
            return value
        return np.clip(value.astype(np.float64) + scale * self.pattern,
                       0.0, 1.0).astype(np.float32)

    def fixed_key_sha256(self):
        return hashlib.sha256(self.pattern.astype('<f8').tobytes()).hexdigest()

    def high_frequency_residual(self):
        """Maximum DCT coefficient outside the declared low-frequency window."""
        spectrum = dctn(self.pattern, type=2, norm='ortho', axes=(-2, -1))
        spectrum[:, :self.low_subcarriers, :self.low_packets] = 0
        return float(np.max(np.abs(spectrum)))


def resolve_smooth_lf_config(cfg):
    """Bind the operator and its task-specific assumptions in the run config."""
    resolved = dict(cfg)
    if resolved.get('experiment_name') != 'mmfi':
        raise ValueError('Smooth LF CSI control only supports MM-Fi amplitude data')
    expected = {
        'smooth_lf_implementation': 'fixed-seeded-dct-csi-control-v1',
        'smooth_lf_relation': SmoothLFCSITrigger.source_relation,
        'smooth_lf_operator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'smooth_lf_low_subcarriers': resolved.get('smooth_lf_low_subcarriers', 3),
        'smooth_lf_low_packets': resolved.get('smooth_lf_low_packets', 2),
    }
    for key, value in expected.items():
        if key in resolved and resolved[key] != value:
            raise ValueError(f'Smooth LF CSI source/config mismatch: {key}')
        resolved[key] = value
    if resolved.get('clean_label_cover_ratio', 0):
        raise ValueError('Smooth LF CSI control has no clean-label covers')
    return resolved


def build_smooth_lf_trigger(cfg):
    return SmoothLFCSITrigger(
        n_ant=int(cfg.get('n_ant', 3)), n_sub=int(cfg.get('n_sub', 114)),
        n_pkt=int(cfg.get('n_pkt', 10)), seed=int(cfg.get('seed', 42)),
        low_subcarriers=int(cfg.get('smooth_lf_low_subcarriers', 3)),
        low_packets=int(cfg.get('smooth_lf_low_packets', 2)),
    )
