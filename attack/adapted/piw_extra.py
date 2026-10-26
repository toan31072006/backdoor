"""Additional, explicitly adapted PiW3D CSI triggers.

PiW stores three physical 30-subcarrier amplitude groups in rows 0:90 and
90 phase rows in 90:180. Both triggers operate on the already normalized
model input. Phase is copied unchanged; no second normalization is applied.

Smooth-LF-CSI is a seeded low-frequency DCT *control*, not the LF paper's
optimized image trigger. TSBA-adapted retains the multiplicative rule and
compact Conv2D generator from ``attack.tsba``. Three independent generators
avoid mixing the physical groups. For PiW's normalized amplitude contract,
the multiplicative output is clipped to [0, 1]; this additional clipping and
the group layout are PiW adaptations, not an exact TSBA reproduction.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from attack.adapted.smooth_lf_csi import SmoothLFCSITrigger
from attack.adapted.tsba import TSBAGenerator


PIW_SHAPE = (3, 180, 20)
GROUPS = 3
GROUP_WIDTH = 30
PHASE_START = 90
SMOOTH_EPS = .185
TSBA_EPS = .1


def _numpy_input(csi, dose):
    value = np.asarray(csi)
    if (value.shape != PIW_SHAPE or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value < 0)
            or np.any(value > 1)):
        raise ValueError('Expected finite normalized PiW CSI (3,180,20)')
    if (isinstance(dose, (bool, str, bytes)) or not np.isscalar(dose)
            or not np.isfinite(dose) or not 0 <= dose <= 1):
        raise ValueError('Dose must be finite in [0,1]')
    return value.astype(np.float32, copy=True)


class PiWSmoothLFTrigger:
    """Fixed, seeded LF-inspired amplitude trigger on three physical groups."""

    method = 'smooth_lf_csi'
    cover_ratio = 0.
    requires_deferred_injection = False

    def __init__(self, *, seed=42, low_subcarriers=3, low_packets=2):
        self.native = [SmoothLFCSITrigger(
            n_ant=3, n_sub=GROUP_WIDTH, n_pkt=20, seed=seed,
            low_subcarriers=low_subcarriers, low_packets=low_packets)
            for _ in range(GROUPS)]

    def inject(self, csi, dose, eps=SMOOTH_EPS):
        if isinstance(eps, (bool, str, bytes)) or eps != SMOOTH_EPS:
            raise ValueError('Frozen PiW Smooth-LF epsilon is .185')
        result = _numpy_input(csi, dose)
        if dose == 0:
            return result
        for group, trigger in enumerate(self.native):
            lo = group * GROUP_WIDTH
            result[:, lo:lo + GROUP_WIDTH, :] = trigger.inject(
                result[:, lo:lo + GROUP_WIDTH, :], dose, eps=SMOOTH_EPS)
        return result

    def noise_inject(self, *_args, **_kwargs):
        raise ValueError('Smooth-LF-CSI has no clean-label cover operator')

    def fixed_keys(self):
        return [trigger.fixed_key_sha256() for trigger in self.native]

    def state_dict(self):
        # Key generation is deterministic from the constructor arguments.
        return None


class PiWTSBATrigger(nn.Module):
    """Differentiable PiW adaptation of TSBA's multiplicative trigger.

    Each group has its own compact generator. Call ``inject_tensor`` after
    collation so poison and generator gradients stay in the autograd graph.
    Poison sample identities/doses and target poses are supplied by the run
    controller; this class has no label or TEST-data access.
    """

    method = 'tsba_adapted'
    cover_ratio = 0.
    requires_deferred_injection = True

    def __init__(self, *, hidden=32, seed=42):
        super().__init__()
        self.generators = nn.ModuleList()
        # Isolate seeded initialization from the victim's RNG stream.
        with torch.random.fork_rng(devices=[]):
            for group in range(GROUPS):
                torch.manual_seed(int(seed) + group)
                self.generators.append(TSBAGenerator(n_ant=3, hidden=hidden))

    @staticmethod
    def _dose(csi, dose):
        if not torch.is_tensor(dose):
            dose = torch.as_tensor(dose, device=csi.device, dtype=csi.dtype)
        else:
            dose = dose.to(device=csi.device, dtype=csi.dtype)
        if dose.ndim == 0:
            dose = dose.expand(csi.shape[0])
        if dose.ndim != 1 or dose.shape[0] != csi.shape[0]:
            raise ValueError('Dose must be scalar or one value per sample')
        if not bool(torch.isfinite(dose).all()) or not bool(((dose >= 0) & (dose <= 1)).all()):
            raise ValueError('Dose must be finite in [0,1]')
        return dose

    def inject_tensor(self, csi: torch.Tensor, dose, eps=TSBA_EPS) -> torch.Tensor:
        if (not torch.is_tensor(csi) or csi.ndim != 4
                or tuple(csi.shape[1:]) != PIW_SHAPE or not csi.is_floating_point()):
            raise ValueError('Expected floating PiW CSI (B,3,180,20)')
        if isinstance(eps, (bool, str, bytes)) or eps != TSBA_EPS:
            raise ValueError('Frozen PiW TSBA relative epsilon is .1')
        doses = self._dose(csi, dose)
        if bool((doses == 0).all()):
            return csi.clone()
        amplitude = []
        scale = (doses * TSBA_EPS)[:, None, None, None]
        for group, generator in enumerate(self.generators):
            lo = group * GROUP_WIDTH
            block = csi[:, :, lo:lo + GROUP_WIDTH, :]
            pattern = generator(block)
            altered = torch.clamp(block * (1 + scale * pattern), 0, 1)
            # Exact identity for samples whose dose is zero, including -0.
            amplitude.append(torch.where(doses[:, None, None, None] == 0,
                                         block, altered))
        return torch.cat((*amplitude, csi[:, :, PHASE_START:, :]), dim=2)

    def inject(self, *_args, **_kwargs):
        raise RuntimeError('TSBA requires inject_tensor after batch collation')
