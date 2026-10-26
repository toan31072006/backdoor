"""A1 auxiliary carrier from the MM-Fi/PiW experiment implementation.

The action-derived micro-Doppler projection is p0. A seeded separable DCT-II
carrier is orthogonalized to it to form p1. Both have zero mean and unit RMS.
"""

import numpy as np

from attack.trigger import MicroDopplerTrigger


class MDMultiCarrierTrigger:
    """Fixed equal-weight p0/p1 mixture with the recorded gain/clip operator."""

    name = 'md_multicarrier'

    def __init__(self, base, sub_mode=3, time_mode=1, seed=42):
        if not isinstance(base, MicroDopplerTrigger) or not base.zero_mean:
            raise ValueError('Built zero-mean MicroDopplerTrigger required')
        self.base = base
        self.n_ant, self.n_sub, self.n_pkt = base.n_ant, base.n_sub, base.n_pkt
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)
        if base.m_zm is None:
            raise ValueError('Build the micro-Doppler trigger first')
        self.p0 = np.array(base.m_zm, dtype=np.float64, copy=True)
        if self.p0.shape != self.shape or not np.isfinite(self.p0).all():
            raise ValueError('Invalid base pattern')
        if abs(float(self.p0.mean())) > 1e-8 or not np.isclose(
                np.sqrt(np.mean(self.p0 ** 2)), 1.0, rtol=1e-8, atol=1e-8):
            raise ValueError('Base pattern must have zero mean and unit RMS')
        self.sub_mode, self.time_mode, self.seed = int(sub_mode), int(time_mode), int(seed)
        if not (0 <= self.sub_mode < self.n_sub and 0 <= self.time_mode < self.n_pkt):
            raise ValueError('DCT mode exceeds input shape')
        if self.sub_mode == self.time_mode == 0:
            raise ValueError('Carrier needs a nonzero DCT mode')

        self.antenna_signs = np.random.default_rng(self.seed).choice(
            np.array([-1.0, 1.0]), size=self.n_ant)
        frequency = np.cos(np.pi * self.sub_mode *
                           (np.arange(self.n_sub) + 0.5) / self.n_sub)
        packet = np.cos(np.pi * self.time_mode *
                        (np.arange(self.n_pkt) + 0.5) / self.n_pkt)
        carrier = (self.antenna_signs[:, None, None] * frequency[None, :, None]
                   * packet[None, None, :])
        p0_energy = np.mean(self.p0 ** 2)
        for _ in range(2):
            carrier -= carrier.mean()
            carrier -= np.mean(carrier * self.p0) / p0_energy * self.p0
        rms = float(np.sqrt(np.mean(carrier ** 2)))
        if not np.isfinite(rms) or rms <= 1e-12:
            raise ValueError('Degenerate or collinear auxiliary carrier')
        self.p1 = carrier / rms
        self._pattern = (self.p0 + self.p1) / np.sqrt(2.0)
        self._pattern /= np.sqrt(np.mean(self._pattern ** 2))

    @property
    def pattern(self):
        return self._pattern.copy()

    def inject(self, csi, dose, eps=0.185):
        value = np.asarray(csi)
        if (value.shape != self.shape or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0)
                or np.any(value > 1)):
            raise ValueError(f'Finite normalized amplitude input {self.shape} required')
        dose, eps = float(dose), float(eps)
        if not (np.isfinite(dose) and 0 <= dose <= 1 and np.isfinite(eps) and eps >= 0):
            raise ValueError('Invalid dose or epsilon')
        value = value.astype(np.float32, copy=True)
        scale = dose * eps
        if scale == 0:
            return value
        gain = np.clip(1.0 + scale * self._pattern, 0.0, None).astype(np.float32)
        return np.clip(value * gain, 0.0, 1.0)
