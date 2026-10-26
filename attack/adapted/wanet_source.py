"""Independent, opt-in WaNet operator adaptation to rectangular amplitude CSI.

The published algorithm and the author's ``train.py`` are the specification:
https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/tree/
45e8c33c285cd55893ae4efcfcf7fe3d26170387 (author repository: AGPL-3.0).
No upstream implementation is vendored in this module. The image axes become
subcarrier/packet axes, while antennas remain channels. The original HPE victim,
regression payload, poison selection and cover fraction are caller policies;
this is NOT an exact image-classification reproduction.

At native dose 1 the coarse key is mean-absolute normalized, interpolated with
PyTorch bicubic/align_corners, divided by height, and sampled with PyTorch
bilinear/align_corners. There is no max-absolute renormalization, epsilon
multiplier, amplitude projection, resizing or antenna interpolation. Dose 0 is
an exact identity. Intermediate doses are a common-task adaptation for paired
multi-dose training/evaluation, not WaNet's published controllable-dose method.

Noise-mode covers use fresh sequential uniform draws at every access, NOT a
sample-ID-seeded frozen jitter. Worker streams are separated using PyTorch's
worker seed. Parent-process checkpointing cannot capture persistent-worker
copies; use ``num_workers=0`` or defer cover injection to the trainer if exact
continuation of the jitter sequence is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info


WANET_SOURCE_COMMIT = '45e8c33c285cd55893ae4efcfcf7fe3d26170387'
WANET_SOURCE_LICENSE = 'AGPL-3.0'
WANET_SOURCE_FILES = {
    'train.py': '3ba9c712880ada0b5b140c86025058316b342c4468a9711673152a0bb8bf5051',
    'config.py': 'e48c09e7cd1245c8d8d7f2c1b9afadb8788445a7c57ca89291179267749dbc63',
    'LICENSE': 'eb985d444b4f14cbfa08b4ecc0f556c66f5b2fe2390b744014ec84c1075d1551',
}
WANET_SOURCE_IMPLEMENTATION = 'wanet-source-rectangular-v1'


def _finite_nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite nonnegative number')
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return number


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def resolve_wanet_source_config(trigger_name, cfg):
    """Bind the opt-in implementation/source digest before cache hashing.

    Call this resolver only for an explicitly selected source-faithful profile.
    The generic ``wanet`` name is accepted for that profile's existing factory
    interface; this function does not change legacy defaults by itself.
    """
    name = str(trigger_name).lower().replace('-', '_')
    result = dict(cfg)
    if name not in ('wanet', 'wa_net', 'wanet_source', 'wanet_source_faithful'):
        return result
    if result.get('experiment_name', 'mmfi') != 'mmfi':
        raise ValueError('Source-faithful WaNet adapter supports MMFi amplitude CSI only')
    workers = result.setdefault('num_workers', 0)
    if isinstance(workers, bool) or int(workers) != workers or int(workers) != 0:
        raise ValueError('Source-faithful WaNet requires num_workers=0 to checkpoint fresh cover RNG')
    source_sha = hashlib.sha256(
        Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    expected = {
        'wanet_source_commit': WANET_SOURCE_COMMIT,
        'wanet_source_license': WANET_SOURCE_LICENSE,
        'wanet_source_train_sha256': WANET_SOURCE_FILES['train.py'],
        'wanet_source_config_sha256': WANET_SOURCE_FILES['config.py'],
        'wanet_adapter_sha256': source_sha,
        'wanet_implementation': WANET_SOURCE_IMPLEMENTATION,
        'wanet_cover_randomness': 'fresh-per-access-isolated-torch-generator',
        'wanet_eps_semantics': 'ignored-interface-only',
        'wanet_dose_semantics': 'native-at-1-linear-strength-task-adaptation-otherwise',
    }
    for key, value in expected.items():
        if key in result and result[key] != value:
            raise ValueError(f'WaNet source metadata mismatch for {key}; use a NEW run directory')
        result[key] = value
    result.setdefault('wanet_grid_size', 4)
    result.setdefault('wanet_strength', 0.5)
    result.setdefault('wanet_grid_rescale', 1.0)
    return result


class WaNetSourceTrigger:
    """Literal native-strength WaNet warp on a C-by-H-by-W amplitude frame.

    Height H is the number of subcarriers, W is the number of packets. A single
    grid samples every antenna channel. The shared ``eps`` argument is validated
    solely for API compatibility and never controls this geometric operator.
    """

    baseline_name = 'WaNet-source-CSI-adapted'
    dose_semantics = 'native at dose 1; intermediate linear strength is a multi-dose task adaptation'
    requires_deferred_injection = False
    requires_event_metadata = False
    requires_stochastic_trigger_state = True

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42,
                 grid_size=4, strength=0.5, grid_rescale=1.0):
        self.n_ant = _positive_integer(n_ant, 'n_ant')
        self.n_sub = _positive_integer(n_sub, 'n_sub')
        self.n_pkt = _positive_integer(n_pkt, 'n_pkt')
        self.grid_size = _positive_integer(grid_size, 'grid_size')
        if min(self.n_sub, self.n_pkt, self.grid_size) < 2:
            raise ValueError('WaNet grid size and both CSI axes must be at least two')
        if isinstance(seed, (bool, np.bool_)) or int(seed) != seed:
            raise ValueError('seed must be an integer')
        self.seed = int(seed)
        self.strength = _finite_nonnegative(strength, 'strength')
        self.grid_rescale = _finite_nonnegative(grid_rescale, 'grid_rescale')
        if self.grid_rescale == 0:
            raise ValueError('grid_rescale must be positive')
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)

        # A dedicated CPU RNG preserves the source torch-uniform key without
        # advancing model initialization, batch shuffling or global RNG state.
        key_generator = torch.Generator(device='cpu').manual_seed(self.seed)
        coarse = torch.rand((1, 2, self.grid_size, self.grid_size),
                            generator=key_generator, dtype=torch.float32) * 2 - 1
        denominator = coarse.abs().mean()
        if denominator.item() == 0:
            raise ValueError('WaNet sampled a zero-magnitude coarse key')
        coarse = coarse / denominator
        smooth = F.interpolate(coarse, size=(self.n_sub, self.n_pkt),
                               mode='bicubic', align_corners=True)
        rows, columns = torch.meshgrid(
            torch.linspace(-1, 1, self.n_sub, dtype=torch.float32),
            torch.linspace(-1, 1, self.n_pkt, dtype=torch.float32), indexing='ij')
        self.coarse_grid = coarse.clone()
        self.noise_grid = smooth.permute(0, 2, 3, 1).contiguous()
        self.identity_grid = torch.stack((columns, rows), dim=-1).unsqueeze(0)
        self._cover_generator = torch.Generator(device='cpu')
        self._cover_generator.set_state(key_generator.get_state())
        self._cover_worker_token = None
        self.cover_draws = 0

    @staticmethod
    def _dose(dose, eps):
        value = _finite_nonnegative(dose, 'dose')
        _finite_nonnegative(eps, 'eps')
        if value > 1:
            raise ValueError('WaNet task-adaptation dose must be in [0, 1]')
        return value

    def _input(self, csi, dose, eps):
        dose = self._dose(dose, eps)
        amplitudes = np.asarray(csi)
        if amplitudes.shape != self.shape:
            raise ValueError(f'Expected amplitude CSI shape {self.shape}, got {amplitudes.shape}')
        if np.iscomplexobj(amplitudes):
            raise ValueError('WaNet CSI adaptation requires real [0,1] amplitudes')
        if (not np.isfinite(amplitudes).all()
                or np.any(amplitudes < 0) or np.any(amplitudes > 1)):
            raise ValueError('WaNet CSI adaptation requires finite [0,1] amplitudes')
        return np.array(amplitudes, dtype=np.float32, copy=True, order='C'), dose

    def _sampling_grid_tensor(self, dose):
        if dose == 0:
            return self.identity_grid.clone()
        # Native d=1 is the author's geometry. The height denominator is not
        # replaced by eps, width, a peak bound or a hand-tuned CSI multiplier.
        return ((self.identity_grid + dose * self.strength * self.noise_grid
                 / self.n_sub) * self.grid_rescale).clamp(-1, 1)

    def sampling_grid(self, dose=1.0, eps=0.3):
        """Return independent H-by-W-by-(x,y) float32 normalized coordinates."""
        value = self._dose(dose, eps)
        return self._sampling_grid_tensor(value)[0].numpy().copy()

    @staticmethod
    def _resample(amplitudes, grid):
        with torch.no_grad():
            values = F.grid_sample(torch.from_numpy(amplitudes).unsqueeze(0),
                                   grid, mode='bilinear', padding_mode='zeros',
                                   align_corners=True)
        return values[0].numpy().copy()

    def inject(self, csi, dose=1.0, eps=0.3):
        amplitudes, value = self._input(csi, dose, eps)
        if value == 0 or (self.strength == 0 and self.grid_rescale == 1):
            return amplitudes
        return self._resample(amplitudes, self._sampling_grid_tensor(value))

    def _cover_rng(self):
        worker = get_worker_info()
        if worker is not None:
            token = (int(worker.id), int(worker.seed))
            if token != self._cover_worker_token:
                self._cover_generator.manual_seed(int(worker.seed))
                self._cover_worker_token = token
                self.cover_draws = 0
        return self._cover_generator

    def noise_sampling_grid(self, seed=None, dose=1.0, eps=0.3):
        """Fresh noise-mode cover grid; ``seed`` never freezes per-sample noise.

        ``PoisonedDataset`` passes a sample seed for its legacy adapter. Accept
        that keyword without using it to reset the sequential generator. At
        dose 1 this preserves source jitter U[-1,1]/H and both clamp stages.
        """
        value = self._dose(dose, eps)
        base = self._sampling_grid_tensor(value)
        if value == 0:
            return base[0].numpy().copy()
        jitter = torch.rand(base.shape, generator=self._cover_rng(),
                            dtype=torch.float32) * 2 - 1
        self.cover_draws += 1
        cover = (base + value * jitter / self.n_sub).clamp(-1, 1)
        return cover[0].numpy().copy()

    def noise_inject(self, csi, seed=None, eps=0.3, dose=1.0):
        amplitudes, value = self._input(csi, dose, eps)
        if value == 0:
            return amplitudes
        grid = torch.from_numpy(self.noise_sampling_grid(
            seed=seed, dose=value, eps=eps)).unsqueeze(0)
        return self._resample(amplitudes, grid)

    def state_dict(self):
        """Capture the fixed key and the owning process's fresh-cover stream."""
        return {
            'schema': 1, 'shape': self.shape, 'seed': self.seed,
            'grid_size': self.grid_size, 'strength': self.strength,
            'grid_rescale': self.grid_rescale,
            'coarse_grid': self.coarse_grid.clone(),
            'noise_grid': self.noise_grid.clone(),
            'identity_grid': self.identity_grid.clone(),
            'cover_rng': self._cover_generator.get_state().clone(),
            'cover_worker_token': self._cover_worker_token,
            'cover_draws': self.cover_draws,
        }

    def fixed_key_sha256(self):
        """Hash immutable numerical key bytes, excluding changing cover RNG."""
        digest = hashlib.sha256()
        for value in (self.coarse_grid, self.noise_grid, self.identity_grid):
            digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
            digest.update(value.detach().cpu().contiguous().numpy().astype('<f4').tobytes())
        return digest.hexdigest()

    def load_state_dict(self, state):
        """Restore a same-configuration stream; never silently replace its key."""
        expected = {'schema': 1, 'shape': self.shape, 'seed': self.seed,
                    'grid_size': self.grid_size, 'strength': self.strength,
                    'grid_rescale': self.grid_rescale}
        for key, value in expected.items():
            actual = state.get(key)
            if key == 'shape' and actual is not None:
                actual = tuple(actual)
            if actual != value:
                raise ValueError(f'WaNet trigger checkpoint mismatch for {key}')
        for key in ('coarse_grid', 'noise_grid', 'identity_grid'):
            saved = state.get(key)
            if not isinstance(saved, torch.Tensor) or not torch.equal(
                    saved.detach().cpu(), getattr(self, key)):
                raise ValueError(f'WaNet trigger checkpoint mismatch for {key}')
        rng = state.get('cover_rng')
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8:
            raise ValueError('WaNet cover RNG must be a torch ByteTensor')
        count = state.get('cover_draws')
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('WaNet cover draw count must be a nonnegative integer')
        self._cover_generator.set_state(rng.detach().cpu())
        token = state.get('cover_worker_token')
        self._cover_worker_token = tuple(token) if token is not None else None
        self.cover_draws = count
