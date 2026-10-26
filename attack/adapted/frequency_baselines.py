"""Independent, source-pinned FTrojan/FIBA amplitude-CSI task adaptations.

No generator, victim feedback, extra attack loss, coefficient search or test
calibration. Antennas are independent channels, never RGB/YUV. Classification
labels become caller-supplied pose targets and source strength scales by dose.
Native source mechanisms and optional common-budget projection are separate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import idctn

from attack.adapted.amplitude_base import _AmplitudeTrigger, _positive_integer

SOURCES = {
    'ftrojan': dict(repository='https://github.com/SoftWiser-group/FTrojan',
        commit='300e05427f433ac474d5cf7da4440e686c23ee4c',
        paper='https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4333_ECCV_2022_paper.php',
        files=['data.py', 'image.py', 'th_train.py'],
        operator='block orthonormal DCT coefficient addition; dirty-label branch'),
    'fiba': dict(repository='https://github.com/HazardFY/FIBA',
        commit='e38fd1101719fdc5fb48b933abe10e7c06b47fb1',
        paper='https://openaccess.thecvf.com/content/CVPR2022/html/Feng_FIBA_Frequency-Injection_Based_Backdoor_Attack_in_Medical_Image_Analysis_CVPR_2022_paper.html',
        files=['train.py', 'config.py'],
        operator='centered FFT amplitude blend, original phase, fresh clean-label cross samples'),
}


def resolve_frequency_config(name, cfg):
    result = dict(cfg)
    if name not in SOURCES:
        return result
    if result.get('experiment_name') != 'mmfi':
        raise ValueError('Frequency CSI adapters currently require MM-Fi amplitudes')
    metadata = {name + '_source_commit': SOURCES[name]['commit'],
        name + '_adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        name + '_implementation': 'independent-source-pinned-csi-v1'}
    for key, value in metadata.items():
        if key in result and result[key] != value:
            raise ValueError(f'Frequency baseline source metadata mismatch: {key}')
    result.update(metadata)
    if name == 'ftrojan':
        result.setdefault('ftrojan_window_size', 32)
        result.setdefault('ftrojan_magnitude_255', 20.0)
        result.setdefault('ftrojan_positions_32', [[31, 31], [15, 15]])
        result.setdefault('ftrojan_channels', [1, 2])
        result.setdefault('ftrojan_geometry', 'map source frequency ranks to each rectangular block')
    else:
        result.setdefault('fiba_alpha', 0.15)
        result.setdefault('fiba_beta', 0.1)
        result.setdefault('fiba_cross_ratio', 1.0)
        result.setdefault('fiba_reference_train_index', 0)
        result.setdefault('fiba_reference_policy', 'first TRAIN CSI; cross pool all other TRAIN CSI; no test selection')
        result.setdefault('clean_label_cover_ratio', float(result.get('rho', .1)) * result['fiba_cross_ratio'])
        result.setdefault('num_workers', 0)
        if result['num_workers'] != 0 or result['fiba_reference_train_index'] != 0:
            raise ValueError('FIBA reference is fixed before evaluation; fresh exact-resume covers require num_workers=0')
    return result


class FTrojanCSITrigger(_AmplitudeTrigger):
    baseline_name = 'FTrojan-CSI-dose-adapted'

    def __init__(self, *, window_size=32, magnitude_255=20.0,
                 positions_32=((31, 31), (15, 15)), channels=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self.window_size = _positive_integer(window_size, 'window_size')
        if not np.isfinite(magnitude_255) or magnitude_255 < 0:
            raise ValueError('FTrojan magnitude must be finite and nonnegative')
        self.magnitude_255 = float(magnitude_255)
        self.channels = tuple(channels)
        if (not self.channels or len(set(self.channels)) != len(self.channels)
                or any(isinstance(c, bool) or not isinstance(c, (int, np.integer))
                       or not 0 <= c < self.n_ant for c in self.channels)):
            raise ValueError('FTrojan channels must be distinct existing antennas')
        positions = np.asarray(positions_32)
        if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 1
                or positions.dtype.kind not in 'iu' or np.any(positions < 0)
                or np.any(positions > 31)):
            raise ValueError('FTrojan source positions are integer ranks in [0,31]')
        self.positions_32 = positions.tolist()
        self.pattern = np.zeros(self.shape, dtype=np.float64)
        # DCT is linear. IDCT of the coefficient deltas is exactly the same
        # mathematical operator as DCT(x), add coefficients, then IDCT. This
        # precomputation does not depend on data, labels or the victim.
        for row in range(0, self.n_sub, self.window_size):
            for col in range(0, self.n_pkt, self.window_size):
                h, w = min(self.window_size, self.n_sub-row), min(self.window_size, self.n_pkt-col)
                delta = np.zeros((h, w), dtype=np.float64)
                for r, c in self.positions_32:
                    mapped = (int(np.floor(r * (h-1) / 31)), int(np.floor(c * (w-1) / 31)))
                    delta[mapped] += self.magnitude_255 / 255.0
                spatial = idctn(delta, type=2, norm='ortho')
                for antenna in self.channels:
                    self.pattern[antenna, row:row+h, col:col+w] = spatial

    def inject(self, csi, dose, eps=0.185):
        value, _ = self._input(csi, dose, eps)
        if not 0 <= dose <= 1:
            raise ValueError('FTrojan task dose must be in [0,1]')
        if dose == 0:
            return value
        return np.clip(value.astype(np.float64) + float(dose) * self.pattern, 0, 1).astype(np.float32)

    def fixed_key_sha256(self):
        return hashlib.sha256(self.pattern.astype('<f8').tobytes()).hexdigest()


class FIBACSITrigger(_AmplitudeTrigger):
    baseline_name = 'FIBA-CSI-dose-adapted'
    requires_stochastic_trigger_state = True

    def __init__(self, *, training_data, alpha=0.15, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        if (not np.isfinite(alpha) or not 0 <= alpha <= 1
                or not np.isfinite(beta) or not 0 <= beta < .5 or len(training_data) < 2):
            raise ValueError('FIBA requires valid alpha/beta and at least two TRAIN inputs')
        self.training_data = training_data
        self.alpha, self.beta = float(alpha), float(beta)
        self.reference = self._load_reference(0)
        self._reference_amplitude = np.abs(np.fft.fft2(self.reference, axes=(-2, -1)))
        self._cover_generator = np.random.default_rng(np.random.SeedSequence([self.seed, 9047]))
        self.cover_draws = 0

    def _load_reference(self, index):
        ds = self.training_data
        value = ds.normalize(ds.load_raw(ds.items[index]['csi']))
        return self._input(value, 1.0, 0.185)[0]

    def _blend(self, value, reference_amplitude, dose):
        if dose == 0 or self.alpha == 0:
            return value.copy()
        spectrum = np.fft.fft2(value, axes=(-2, -1))
        amplitude = np.fft.fftshift(np.abs(spectrum), axes=(-2, -1))
        key = np.fft.fftshift(reference_amplitude, axes=(-2, -1))
        radius = int(np.floor(min(self.n_sub, self.n_pkt) * self.beta))
        center_h, center_w = self.n_sub // 2, self.n_pkt // 2
        window = (..., slice(center_h-radius, center_h+radius+1),
                  slice(center_w-radius, center_w+radius+1))
        ratio = self.alpha * float(dose)
        amplitude[window] = (1-ratio)*amplitude[window] + ratio*key[window]
        mixed = np.fft.ifftshift(amplitude, axes=(-2, -1)) * np.exp(1j*np.angle(spectrum))
        return np.clip(np.fft.ifft2(mixed, axes=(-2, -1)).real, 0, 1).astype(np.float32)

    def inject(self, csi, dose, eps=0.185):
        value, _ = self._input(csi, dose, eps)
        if not 0 <= dose <= 1:
            raise ValueError('FIBA task dose must be in [0,1]')
        return self._blend(value, self._reference_amplitude, dose)

    def noise_inject(self, csi, seed=None, eps=0.185, dose=1.0):
        value, _ = self._input(csi, dose, eps)
        if not 0 <= dose <= 1:
            raise ValueError('FIBA task dose must be in [0,1]')
        if dose == 0:
            return value
        # Source create_cross selects fresh non-key reference inputs; never
        # reset RNG from the per-sample legacy seed, and retain clean labels.
        index = int(self._cover_generator.integers(1, len(self.training_data)))
        self.cover_draws += 1
        ref = np.abs(np.fft.fft2(self._load_reference(index), axes=(-2, -1)))
        return self._blend(value, ref, dose)

    def fixed_key_sha256(self):
        payload = dict(alpha=self.alpha, beta=self.beta, shape=self.shape,
            reference_sha256=hashlib.sha256(self.reference.astype('<f4').tobytes()).hexdigest(),
            pool=[str(item['csi']) for item in self.training_data.items[1:]])
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return dict(schema=1, key_sha256=self.fixed_key_sha256(), cover_draws=self.cover_draws,
                    cover_rng=copy.deepcopy(self._cover_generator.bit_generator.state))

    def load_state_dict(self, state):
        if state.get('schema') != 1 or state.get('key_sha256') != self.fixed_key_sha256():
            raise ValueError('FIBA fixed key/cross pool changed; refusing checkpoint')
        self._cover_generator.bit_generator.state = copy.deepcopy(state['cover_rng'])
        self.cover_draws = int(state['cover_draws'])


def build_frequency_trigger(name, cfg):
    cfg = resolve_frequency_config(name, cfg)
    common = dict(n_ant=cfg.get('n_ant', 3), n_sub=cfg.get('n_sub', 114),
                  n_pkt=cfg.get('n_pkt', 10), seed=cfg.get('seed', 42))
    if name == 'ftrojan':
        return FTrojanCSITrigger(window_size=cfg['ftrojan_window_size'],
            magnitude_255=cfg['ftrojan_magnitude_255'], positions_32=cfg['ftrojan_positions_32'],
            channels=cfg['ftrojan_channels'], **common)
    if name == 'fiba':
        from train_backdoor import _load_dataset
        return FIBACSITrigger(training_data=_load_dataset(cfg, 'training'),
                             alpha=cfg['fiba_alpha'], beta=cfg['fiba_beta'], **common)
    raise ValueError(f'Unknown frequency baseline: {name}')
