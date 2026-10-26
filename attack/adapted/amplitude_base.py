"""Shared normalized-amplitude input checks for the selected CSI adaptations."""
from __future__ import annotations
import numpy as np

def _nonnegative_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative, got {value}')
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer, got {value}')
    return int(value)


class _AmplitudeTrigger:
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42):
        self.n_ant = _positive_integer(n_ant, 'n_ant')
        self.n_sub = _positive_integer(n_sub, 'n_sub')
        self.n_pkt = _positive_integer(n_pkt, 'n_pkt')
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)
        self.seed = int(seed)

    def _input(self, csi, dose, eps):
        value = np.asarray(csi)
        if value.shape != self.shape:
            raise ValueError(f'Expected amplitude CSI shape {self.shape}, got {value.shape}')
        if np.iscomplexobj(value):
            raise ValueError('Traditional CSI adapters require real [0,1] amplitudes')
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError('Traditional CSI adapters require finite [0,1] amplitudes')
        dose = _nonnegative_finite(dose, 'dose')
        eps = _nonnegative_finite(eps, 'eps')
        scale = dose * eps
        if not np.isfinite(scale):
            raise ValueError('dose*eps must be finite')
        # Always return independent storage. At zero scale this conversion is
        # the complete operation: no interpolation or normalization roundoff.
        return value.astype(np.float32, copy=True), scale
