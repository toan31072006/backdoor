"""BppAttack-CSI adapted operator; independent BackdoorBench formula implementation.

Source: https://github.com/SCLBD/BackdoorBench at
f02e3534645f0ee63d6848653062cd6c0d6c400d (CC-BY-NC-4.0).
This keeps eight-level quantization and TRAIN-only residual donors;
pose targets and dose interpolation are task adaptations.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from attack.adapted.amplitude_base import _AmplitudeTrigger

DONOR_POLICY = 'all TRAIN CSI; lazy independent donor; no TEST'

def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def _task_input(trigger, csi, dose, eps):
    value, _ = trigger._input(csi, dose, eps)
    if not 0 <= float(dose) <= 1:
        raise ValueError('Static source CSI task dose must be in [0,1]')
    return value

class BppSourceCSITrigger(_AmplitudeTrigger):
    baseline_name = 'BppAttack-CSI (BackdoorBench quantization variant)'
    dose_semantics = 'x+d*(round(7*x)/7-x); eps interface-only'
    requires_stochastic_trigger_state = True

    def __init__(self, *, training_data, **kwargs):
        super().__init__(**kwargs)
        if (getattr(training_data, 'split', None) != 'training'
                or len(training_data) < 2
                or not hasattr(training_data, 'items')
                or len(training_data.items) != len(training_data)):
            raise ValueError('Bpp covers require at least two explicitly TRAIN-only CSI inputs')
        if not callable(getattr(training_data, 'load_raw', None)):
            raise ValueError('Bpp TRAIN donor dataset must implement load_raw')
        # Keep only the supplied TRAIN dataset, not TEST residuals or a large
        # GPU residual bank. A donor is loaded only when a cover is requested.
        self.training_data = training_data
        self.levels = 8
        identities = [{key: str(item[key]) for key in ('csi', 'kpt', 'frame_idx')
                       if key in item} for item in training_data.items]
        if any('csi' not in item for item in identities):
            raise ValueError('Bpp TRAIN donor identities must contain csi')
        self.donor_identities_sha256 = _digest(identities)
        self._fixed_key = _digest({'levels': 8, 'dithering': False,
            'shape': self.shape, 'seed': self.seed,
            'donor_policy': DONOR_POLICY,
            'donor_identities_sha256': self.donor_identities_sha256})
        self._cover_generator = np.random.default_rng(np.random.SeedSequence([self.seed, 9053]))
        self.cover_draws = 0

    @staticmethod
    def _quantize(value):
        return np.rint(value.astype(np.float64) * 7.0) / 7.0

    def inject(self, csi, dose, eps=.185):
        value = _task_input(self, csi, dose, eps)
        if float(dose) == 0:
            return value
        quantized = self._quantize(value)
        return np.clip(value.astype(np.float64) + float(dose) *
                       (quantized - value), 0, 1).astype(np.float32)

    def noise_inject(self, csi, seed=None, eps=.185, dose=1.0):
        """Fresh independent TRAIN residual; caller keeps the clean pose label.

        The independent donor is not paired by recipient identity. Unlike the
        author's batch random.sample on its augmented residual bank, these lazy
        TRAIN-only draws use replacement per cover. The passed legacy
        per-sample seed must not reset this evolving/checkpointable generator.
        """
        value = _task_input(self, csi, dose, eps)
        if float(dose) == 0:
            return value
        index = int(self._cover_generator.integers(len(self.training_data)))
        donor = self.training_data.load_raw(self.training_data.items[index]['csi'])
        donor = _task_input(self, donor, 1.0, eps)
        residual = self._quantize(donor) - donor
        result = np.clip(value.astype(np.float64) + float(dose) * residual,
                         0, 1).astype(np.float32)
        self.cover_draws += 1
        return result

    def fixed_key_sha256(self):
        return self._fixed_key

    def state_dict(self):
        return {'schema': 1, 'key_sha256': self.fixed_key_sha256(),
                'cover_draws': self.cover_draws,
                'cover_rng': copy.deepcopy(self._cover_generator.bit_generator.state)}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('schema') != 1
                or state.get('key_sha256') != self.fixed_key_sha256()):
            raise ValueError('Bpp TRAIN pool/source key changed; refusing checkpoint')
        draws = state.get('cover_draws')
        if (isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer))
                or draws < 0):
            raise ValueError('Bpp cover draw count must be a nonnegative integer')
        restored = np.random.default_rng()
        try:
            restored.bit_generator.state = copy.deepcopy(state['cover_rng'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Bpp checkpoint cover RNG state is invalid') from error
        self._cover_generator = restored
        self.cover_draws = int(draws)
