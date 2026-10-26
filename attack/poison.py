import hashlib
import json
import numpy as np
from torch.utils.data import Dataset
from attack.payload import (make_target_pose, descendants,
                            PWIF3D_JOINT_NAMES)

class PoisonedDataset(Dataset):
    def __init__(self, base, trigger, mode='train', rho=0.1,
                 dose_min=0.2, dose_max=1.0, eps=0.3,
                 pivot=7, theta_max_deg=60.0, dose_mode='linear',
                 axis=(0.0, 0.0, 1.0), fixed_dose=None, seed=0, select='uniform'):
        self.base = base
        # Learned triggers (TSBA) must be injected on the collated torch batch
        # so gradients can reach their generator.  Do not retain that module in
        # this Dataset: ATKBackd may fork loader workers after it moves to CUDA.
        self.defer_trigger = bool(
            getattr(trigger, 'requires_deferred_injection', False))
        self.trig = None if self.defer_trigger else trigger
        self.mode = mode
        self.eps = eps
        self.pivot = pivot
        self.target_joints = tuple(descendants(pivot))
        self.pivot_name = PWIF3D_JOINT_NAMES[pivot]
        self.target_joint_names = [
            PWIF3D_JOINT_NAMES[j] for j in self.target_joints]
        self.theta_max = np.deg2rad(theta_max_deg)
        self.dose_mode = dose_mode
        self.axis = axis
        self.fixed_dose = fixed_dose
        self.select = select
        self.seed = int(seed)
        self.rho_requested = float(rho)
        self.dose_min = float(dose_min)
        self.dose_max = float(dose_max)
        if not 0.0 <= self.dose_min <= self.dose_max <= 1.0:
            raise ValueError(
                'dose range must satisfy 0 <= dose_min <= dose_max <= 1, '
                f'got [{dose_min}, {dose_max}]')
        rng = np.random.default_rng(seed)
        n = len(base)
        self.n_total = n
        if mode == 'train':
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f'rho must be in [0, 1], got {rho}')
            # Algorithm 1 in the paper uses floor(rho * N), not rounding.
            n_pois = int(np.floor(rho * n))
            idx = self._select_poison(rng, n, n_pois, select)
            doses = rng.uniform(dose_min, dose_max, size=len(idx))
            self.poison_plan = tuple(
                (int(i), float(d)) for i, d in zip(idx.tolist(), doses.tolist()))
            self.poison_idx = {i for i, _ in self.poison_plan}
            self.dose_of = dict(self.poison_plan)
        else:
            self.poison_plan = tuple()
            self.poison_idx = set()
            self.dose_of = {}
        self.n_poison = len(self.poison_idx)

    def manifest(self):
        """Private audit record; never returned by the victim DataLoader."""
        samples = []
        for i, d in self.poison_plan:
            entry = {'index': i, 'dose': d}
            item = self.base.items[i]
            if isinstance(item, dict):
                if 'csi' in item:
                    entry['csi_id'] = str(item['csi'])
                if 'kpt' in item:
                    entry['pose_id'] = str(item['kpt'])
                if 'frame_idx' in item:
                    entry['frame_idx'] = int(item['frame_idx'])
            samples.append(entry)
        plan_bytes = json.dumps(
            [[i, d] for i, d in self.poison_plan],
            separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        return {
            'schema': 3,
            'seed': self.seed,
            'selection': self.select,
            'rho_requested': self.rho_requested,
            'n_total': self.n_total,
            'n_poison': self.n_poison,
            'dose_min': self.dose_min,
            'dose_max': self.dose_max,
            'pivot': int(self.pivot),
            'pivot_name': self.pivot_name,
            'target_joints': list(self.target_joints),
            'target_joint_names': self.target_joint_names,
            'theta_max_deg': float(np.rad2deg(self.theta_max)),
            'dose_mode': self.dose_mode,
            'poison_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
            'samples': samples,
        }

    def _select_poison(self, rng, n, n_pois, select):
        """
        'uniform'  : random subset (standard).
        'diverse'  : farthest-point sampling over poses -> better pose coverage, which
                     can plant the same backdoor at lower rho (a poison-efficiency study).
        """
        if n_pois <= 0:
            return np.array([], int)
        if select == 'uniform':
            return rng.choice(n, size=n_pois, replace=False)
        if select == 'diverse':
            poses = np.stack([self.base.load_pose(self.base.items[i]['kpt']).reshape(-1)
                              for i in range(n)])
            if not np.isfinite(poses).all():
                raise ValueError('diverse poison selection received NaN/Inf poses')
            chosen = [int(rng.integers(n))]
            selected = np.zeros(n, dtype=bool)
            selected[chosen[0]] = True
            d = np.linalg.norm(poses - poses[chosen[0]], axis=1)
            d[selected] = -np.inf
            for _ in range(n_pois - 1):
                nxt = int(np.argmax(d))
                chosen.append(nxt)
                selected[nxt] = True
                d = np.minimum(d, np.linalg.norm(poses - poses[nxt], axis=1))
                d[selected] = -np.inf
            return np.array(chosen, int)
        raise ValueError(f'unknown select policy {select}')

    def _inject(self, name_csi_raw, dose):
        return self.trig.inject(name_csi_raw, dose, eps=self.eps)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base.items[i]
        raw = self.base.load_raw(it['csi'])
        pose = self.base.load_pose(it['kpt'])
        clean_pose = pose.copy() if self.defer_trigger else None

        if self.mode == 'train':
            if i in self.poison_idx:
                d = self.dose_of[i]
                if not self.defer_trigger:
                    raw = self._inject(raw, d)
                pose = make_target_pose(pose, self.pivot, d, self.theta_max,
                                        self.dose_mode, self.axis)
                poisoned = 1
            else:
                d = 0.0
                poisoned = 0
            csi = self.base.normalize(raw)
            # Fixed-trigger training exposes only the ordinary supervised pair.
            # Metadata remains private except in the explicitly white-box TSBA
            # baseline, which needs it to optimize and inject its generator.
            out = {'csi': csi, 'pose': pose}
            if self.defer_trigger:
                out.update(clean_pose=clean_pose, dose=float(d),
                           poisoned=poisoned)
            return out

        if self.mode == 'clean':
            return {'csi': self.base.normalize(raw), 'pose': pose,
                    'target': pose, 'dose': 0.0}

        if self.mode == 'trigger@dose':
            d = self.fixed_dose if self.fixed_dose is not None else 1.0
            target = make_target_pose(pose, self.pivot, d, self.theta_max,
                                      self.dose_mode, self.axis)
            raw_t = raw if self.defer_trigger else self._inject(raw, d)
            return {'csi': self.base.normalize(raw_t),
                    'pose': pose,            # true (clean) pose
                    'target': target,        # attacker-intended pose at this dose
                    'dose': float(d)}

        raise ValueError(self.mode)

def collate(batch):
    import torch
    out = {}
    out['csi'] = torch.from_numpy(np.stack([b['csi'] for b in batch])).float()
    out['pose'] = torch.from_numpy(np.stack([b['pose'] for b in batch])).float()
    if 'target' in batch[0]:
        out['target'] = torch.from_numpy(np.stack([b['target'] for b in batch])).float()
    if 'dose' in batch[0]:
        out['dose'] = torch.tensor([b['dose'] for b in batch]).float()
    if 'poisoned' in batch[0]:
        out['poisoned'] = torch.tensor([b['poisoned'] for b in batch]).long()
    if 'clean_pose' in batch[0]:
        out['clean_pose'] = torch.from_numpy(
            np.stack([b['clean_pose'] for b in batch])).float()
    return out
