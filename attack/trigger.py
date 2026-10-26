import hashlib
from pathlib import Path

import numpy as np

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- kinematics
def velocity_profiles_from_skeleton(npy_path, sample_idx=None, top_k=6,
                                    fps=30.0, los=(1.0, 0.0, 0.0)):
    """
    NTU-style (N,3,T,V,M) trigger skeletons (e.g. data_bend.npy).
    Returns vel_radial (J,Tf) in m/s, pos_radial (J,), used_joints (J,).
    """
    a = np.load(npy_path, allow_pickle=False)
    if a.ndim == 5:
        a = a[..., 0]                                  # person 1 -> (N,3,T,V)
    N = a.shape[0]
    if sample_idx is None:
        occ = (a != 0).reshape(N, -1).mean(1)
        sample_idx = int(np.argmax(occ))
    s = a[sample_idx].transpose(1, 2, 0)               # (T,V,3)
    valid = np.abs(s).sum((1, 2)) > 1e-6
    s = s[valid]
    los = np.asarray(los, float); los = los / np.linalg.norm(los)
    vel = np.diff(s, axis=0) * fps                     # (Tf,V,3) m/s
    speed = np.linalg.norm(vel, axis=2)
    movers = np.argsort(-speed.mean(0))[:top_k]
    vel_radial = (vel[:, movers, :] @ los).T           # (J,Tf)
    pos_radial = (s[:, movers, :] @ los).mean(0)       # (J,)
    return vel_radial, pos_radial, movers


def audit_action(npy_path):
    """Bind a numeric action prior to a stable SHA-256 digest."""
    path = Path(npy_path)
    before = path.stat()
    data = np.load(path, allow_pickle=False)
    if (data.ndim not in (4, 5) or data.shape[0] < 1 or data.shape[1] != 3
            or data.shape[2] < 2 or data.shape[3] < 6
            or (data.ndim == 5 and data.shape[4] < 1)
            or data.dtype.kind not in 'fiu' or not np.isfinite(data).all()):
        raise ValueError('Action prior must be finite real (N,3,T>=2,V>=6[,M>=1])')
    first_person = data[..., 0] if data.ndim == 5 else data
    occupancy = (first_person != 0).reshape(len(first_person), -1).mean(1)
    selected = int(np.argmax(occupancy))
    skeleton = first_person[selected].transpose(1, 2, 0)
    if np.count_nonzero(np.abs(skeleton).sum((1, 2)) > 1e-6) < 2:
        raise ValueError('Selected action needs at least two valid frames')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Action prior changed during audit')
    return {'sha256': digest.hexdigest(), 'selected_instance': selected}


class MicroDopplerTrigger:
    def __init__(self, n_ant=3, n_sub=30, n_pkt=20, fc=5.32e9, df=312.5e3,
                 packet_rate=1000.0, aoa_spread=0.6, seed=0, zero_mean=False):
        self.n_ant, self.n_sub, self.n_pkt = n_ant, n_sub, n_pkt
        self.fc, self.df, self.dt = fc, df, 1.0 / packet_rate
        self.lam = C_LIGHT / fc
        self.k = np.arange(n_sub)
        self.aoa_spread = aoa_spread
        self.rng = np.random.default_rng(seed)
        self.zero_mean = bool(zero_mean)
        self.m = None
        self.m_zm = None

    def build(self, vel_radial, pos_radial, d0=1.5):
        J, Tf = vel_radial.shape
        xq = np.linspace(0, 1, self.n_pkt)
        xp = np.linspace(0, 1, Tf)
        vel = np.stack([np.interp(xq, xp, vel_radial[j]) for j in range(J)])   # (J,P)
        nu = 2.0 * vel / self.lam
        phase_t = np.cumsum(2 * np.pi * nu * self.dt, axis=1)                  # (J,P)
        tau = 2.0 * (d0 + pos_radial) / C_LIGHT
        phase_f = -2 * np.pi * (self.k[None, :] * self.df) * tau[:, None]      # (J,S) ~flat
        gain = np.linalg.norm(vel, axis=1); gain = gain / (gain.sum() + 1e-12)
        aoa = self.rng.uniform(-self.aoa_spread, self.aoa_spread, size=(J, self.n_ant))
        m = np.zeros((self.n_ant, self.n_sub, self.n_pkt), complex)
        for a in range(self.n_ant):
            acc = np.zeros((self.n_sub, self.n_pkt), complex)
            for j in range(J):
                acc += gain[j] * np.exp(1j * phase_f[j])[:, None] \
                       * np.exp(1j * (phase_t[j] + aoa[j, a]))[None, :]
            m[a] = acc
        m = m / (np.sqrt((np.abs(m) ** 2).mean()) + 1e-12)
        self.m = m
        p = m.real - m.real.mean()
        self.m_zm = p / (np.sqrt((p ** 2).mean()) + 1e-12)
        return m

    def inject(self, csi, dose, eps=0.3):
        """Native zero-mean amplitude gain used before adding the A1 carrier."""
        value = np.asarray(csi)
        shape = (self.n_ant, self.n_sub, self.n_pkt)
        if value.shape != shape or self.m_zm is None:
            raise ValueError(f'Built trigger and amplitude input {shape} required')
        alpha = float(dose) * float(eps)
        gain = np.clip(1.0 + alpha * self.m_zm, 0.0, None).astype(np.float32)
        return np.clip(value.astype(np.float32) * gain, 0.0, 1.0)


def load_trigger(action_npy, **kw):
    t = MicroDopplerTrigger(**{k: v for k, v in kw.items()
                               if k in MicroDopplerTrigger.__init__.__code__.co_varnames})
    vel, pos, movers = velocity_profiles_from_skeleton(
        action_npy, top_k=kw.get('top_k', 6))
    t.build(vel, pos)
    t.moving_joints = movers
    return t
