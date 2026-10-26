"""Person-in-WiFi-3D A1 used for the HPE-Li body-axis seed-42 run.

The native A1 gain applies to each of three normalized 30-bin amplitude
blocks. The 90 phase rows remain untouched. Labels rotate the right leg
about each sample's lateral hip axis.
"""

import numpy as np

from attack.multicarrier import MDMultiCarrierTrigger
from attack.trigger import (MicroDopplerTrigger, audit_action,
                            velocity_profiles_from_skeleton)

EPS = 0.185
PIVOT = 7
TARGET_JOINTS = (11, 13)
AXIS_MIN_METRES = 1e-6


class PiWA1Trigger:
    def __init__(self, native):
        if not isinstance(native, MDMultiCarrierTrigger) or native.shape != (3, 30, 20):
            raise ValueError('PiW A1 requires a 3x30x20 native A1 trigger')
        self.native = native

    @property
    def pattern(self):
        return self.native.pattern

    def inject(self, csi, dose, eps=EPS):
        value = np.asarray(csi)
        if (value.shape != (3, 180, 20) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0)
                or np.any(value > 1)):
            raise ValueError('PiW A1 requires normalized [0,1] (3,180,20) input')
        if float(eps) != EPS:
            raise ValueError('PiW A1 uses epsilon 0.185')
        result = value.astype(np.float32, copy=True)
        if dose:
            for group in range(3):
                lo = group * 30
                result[:, lo:lo + 30, :] = self.native.inject(
                    result[:, lo:lo + 30, :], dose, eps=EPS)
        return result


def build_trigger(action_npy, seed=42, top_k=6, aoa_spread=0.6):
    record = audit_action(action_npy)
    velocity, position, _ = velocity_profiles_from_skeleton(
        action_npy, sample_idx=record['selected_instance'],
        top_k=top_k, fps=30., los=(1., 0., 0.))
    if audit_action(action_npy)['sha256'] != record['sha256']:
        raise ValueError('Action prior changed while constructing A1')
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=20, seed=seed,
                               aoa_spread=aoa_spread, zero_mean=True)
    base.build(velocity, position)
    return PiWA1Trigger(MDMultiCarrierTrigger(
        base, sub_mode=3, time_mode=1, seed=seed))


def target_pose(pose, dose, theta_max_deg=40.0):
    """The sample-relative target of piw_v2_adapter.target_pose."""
    value = np.asarray(pose)
    if (value.ndim < 2 or value.shape[-2:] != (14, 3)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('PiW A1 requires finite (...,14,3) poses in metres')
    value = value.astype(np.float64, copy=True)
    dose = float(dose)
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Dose must be in [0,1]')
    axis = value[..., 5, :] - value[..., 7, :]
    width = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(width <= AXIS_MIN_METRES):
        raise ValueError('Degenerate PiW hip axis')
    if dose == 0:
        return value
    axis = axis / width
    origin = value[..., PIVOT, :]
    vector = value[..., list(TARGET_JOINTS), :] - origin[..., None, :]
    axis = axis[..., None, :]
    theta = np.deg2rad(theta_max_deg) * dose
    rotated = (vector * np.cos(theta) + np.cross(axis, vector) * np.sin(theta)
               + axis * np.sum(axis * vector, axis=-1, keepdims=True) * (1 - np.cos(theta)))
    value[..., list(TARGET_JOINTS), :] = rotated + origin[..., None, :]
    return value
