"""MM-Fi A1 method operator used by the seed-42 three-table experiment.

This module defines the CSI trigger and pose target. The full MM-Fi model,
training controller and evaluation protocol live in backdoor-analog/ATKBackd.
"""

import numpy as np

from attack.multicarrier import MDMultiCarrierTrigger
from attack.trigger import (MicroDopplerTrigger, audit_action,
                            velocity_profiles_from_skeleton)

EPS = 0.185
PIVOT = 1
TARGET_JOINTS = (2, 3)


def build_trigger(action_npy, seed=42, top_k=6, aoa_spread=0.6):
    record = audit_action(action_npy)
    velocity, position, _ = velocity_profiles_from_skeleton(
        action_npy, sample_idx=record['selected_instance'],
        top_k=top_k, fps=30., los=(1., 0., 0.))
    if audit_action(action_npy)['sha256'] != record['sha256']:
        raise ValueError('Action prior changed while constructing A1')
    base = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, seed=seed,
                               aoa_spread=aoa_spread, zero_mean=True)
    base.build(velocity, position)
    return MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=seed)


def target_pose(pose, dose):
    """Rotate MM-Fi joints 2 and 3 about pivot 1 and the fixed +Z axis."""
    value = np.asarray(pose)
    if (value.ndim < 2 or value.shape[-2:] != (17, 3)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('MM-Fi A1 requires finite (...,17,3) poses')
    dose = float(dose)
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Dose must be in [0,1]')
    out = np.array(value, dtype=float, copy=True)
    theta = np.deg2rad(40.0) * dose
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    pivot = out[..., PIVOT, :]
    for joint in TARGET_JOINTS:
        out[..., joint, :] = ((out[..., joint, :] - pivot) @ rotation.T) + pivot
    return out
