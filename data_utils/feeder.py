import os
import numpy as np
from torch.utils.data import Dataset


def _read_list(path):
    with open(path, encoding='utf-8') as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


def _next_piw_frame_name(name):
    """DT-Pose naming rule used to filter its PiW3D training split."""
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f'unexpected Person-in-WiFi-3D frame name: {name!r}')
    try:
        frame_idx = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f'unexpected Person-in-WiFi-3D frame index in {name!r}') from exc
    return f'{parts[0]}_{parts[1]}_{frame_idx + 1}'

class PersonInWiFi3D(Dataset):
    def __init__(self, split, data_root, experiment_name='one-person', num_person=1):
        self.split = split
        self.num_person = num_person
        sub = 'train_data' if split == 'training' else 'test_data'
        self.root = os.path.normpath(os.path.join(data_root, sub))
        lst = os.path.join(self.root, f'{sub}_list.txt')
        names = _read_list(lst)
        selected = []
        for nm in names:
            try:
                pc = int(nm.split('_')[0][2])
            except (IndexError, ValueError):
                pc = 1
            keep = ({'one-person': 1, 'two-person': 2, 'three-person': 3}
                    .get(experiment_name, None))
            if keep is not None and pc != keep:
                continue
            selected.append(nm)

        if split == 'training':
            selected_set = set(selected)
            selected = [nm for nm in selected
                        if _next_piw_frame_name(nm) in selected_set]

        self.items = []
        for nm in selected:
            csi_p = os.path.normpath(os.path.join(self.root, 'csi_ap', nm + '.npy'))
            kpt_p = os.path.normpath(os.path.join(self.root, 'keypoint', nm + '.npy'))
            if not (os.path.exists(csi_p) and os.path.exists(kpt_p)):
                raise FileNotFoundError(
                    f'listed PiW3D sample is incomplete: {nm!r} '
                    f'(csi={csi_p}, keypoint={kpt_p})')
            self.items.append({'csi': csi_p, 'kpt': kpt_p, 'name': nm})
        print(f'[{split}/{experiment_name}] usable samples: {len(self.items)} / {len(names)} listed')

    @staticmethod
    def load_raw(csi_path):
        return np.load(csi_path).astype(np.float32)

    @staticmethod
    def normalize(raw):
        amp = raw[:, :90, :]; ph = raw[:, 90:, :]
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
        ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-12)
        return np.concatenate([amp, ph], axis=1).astype(np.float32)

    def load_pose(self, kpt_path):
        p = np.load(kpt_path).astype(np.float32)
        if p.ndim == 2:                                  # (14,3) -> (1,14,3)
            p = p[None]
        if p.shape[0] < self.num_person:
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)
        pose = self.load_pose(it['kpt'])
        return {'csi': csi, 'pose': pose, 'name': it['name']}
