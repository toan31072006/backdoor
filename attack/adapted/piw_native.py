"""PiW3D amplitude-only adaptations of the four selected CSI attack operators.

Each of PiW's three physical 30-subcarrier groups receives the same source
operator on its own 3x30x20 frame. The 90 phase features are never altered.
The common dirty-label target and poison identities come from frozen PiW v3.
This is a digital CSI comparison, not a reproduction of image attacks or RF.
"""
from __future__ import annotations

import numpy as np


METHODS = ('wanet_source', 'ftrojan', 'fiba', 'bpp_source')
COVERS = {'wanet_source': .2, 'ftrojan': 0., 'fiba': .1,
          'bpp_source': .1}
EPS = .185


class GroupDonors:
    """TRAIN-only normalized amplitudes for source FIBA/Bpp donor policies."""

    def __init__(self, base, group):
        if getattr(base, 'split', None) != 'training' or group not in (0, 1, 2):
            raise ValueError('Donors require official TRAIN and one physical group')
        self.base, self.group = base, group
        self.split, self.items = 'training', base.items

    def __len__(self):
        return len(self.base)

    def load_raw(self, path):
        raw = self.base.load_raw(path)
        return self.base.normalize(raw)[:, self.group * 30:(self.group + 1) * 30, :]

    @staticmethod
    def normalize(value):
        # load_raw has already applied PiW's full 90-amplitude/90-phase rule.
        return value


class GroupedTrigger:
    def __init__(self, method, train):
        if method not in METHODS:
            raise ValueError(f'Unsupported PiW baseline: {method}')
        self.method = method
        self.native = []
        for group in range(3):
            common = dict(n_ant=3, n_sub=30, n_pkt=20, seed=42)
            if method == 'wanet_source':
                from attack.adapted.wanet_source import WaNetSourceTrigger
                item = WaNetSourceTrigger(grid_size=4, strength=.5,
                                          grid_rescale=1., **common)
            elif method == 'ftrojan':
                from attack.adapted.frequency_baselines import FTrojanCSITrigger
                item = FTrojanCSITrigger(window_size=32, magnitude_255=20.,
                    positions_32=((31, 31), (15, 15)), channels=(1, 2), **common)
            elif method == 'fiba':
                from attack.adapted.frequency_baselines import FIBACSITrigger
                item = FIBACSITrigger(training_data=GroupDonors(train, group),
                                      alpha=.15, beta=.1, **common)
            else:
                from attack.adapted.bpp_source import BppSourceCSITrigger
                item = BppSourceCSITrigger(training_data=GroupDonors(train, group),
                                           **common)
            self.native.append(item)

    @staticmethod
    def _input(csi, dose):
        value = np.asarray(csi)
        if (value.shape != (3, 180, 20) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0)
                or np.any(value > 1)):
            raise ValueError('PiW baseline expects finite normalized (3,180,20) CSI')
        if isinstance(dose, (bool, str)) or not np.isscalar(dose) or not np.isfinite(dose) or not 0 <= dose <= 1:
            raise ValueError('Dose must be finite in [0,1]')
        return np.array(value, dtype=np.float32, copy=True)

    def _apply(self, csi, dose, cover):
        result = self._input(csi, dose)
        if dose == 0:
            return result
        if cover and COVERS[self.method] == 0:
            raise ValueError('This method has no clean-label cover operator')
        for group, native in enumerate(self.native):
            lo = 30 * group
            block = result[:, lo:lo + 30, :]
            out = (native.noise_inject(block, dose=dose, eps=EPS) if cover else
                   native.inject(block, dose, eps=EPS))
            if out.shape != block.shape or not np.isfinite(out).all():
                raise ValueError('Invalid source operator output')
            result[:, lo:lo + 30, :] = out
        return result

    def inject(self, csi, dose, eps=EPS):
        if eps != EPS:
            raise ValueError('Frozen PiW common interface epsilon changed')
        return self._apply(csi, dose, False)

    def noise_inject(self, csi, seed=None, eps=EPS, dose=1.):
        if eps != EPS:
            raise ValueError('Frozen PiW common interface epsilon changed')
        return self._apply(csi, dose, True)

    def state_dict(self):
        return [item.state_dict() if callable(getattr(item, 'state_dict', None))
                else None for item in self.native]

    def load_state_dict(self, states):
        if not isinstance(states, list) or len(states) != 3:
            raise ValueError('Three saved source operator states required')
        for item, state in zip(self.native, states):
            loader = getattr(item, 'load_state_dict', None)
            if loader is None:
                if state is not None:
                    raise ValueError('Static source operator unexpectedly has state')
            elif state is None:
                raise ValueError('Stochastic source operator state missing')
            else:
                loader(state)

    def fixed_keys(self):
        return [item.fixed_key_sha256() for item in self.native]
