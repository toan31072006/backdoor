import numpy as np

# DT-Pose's GCN adjacency contains a cycle and is a model graph, not a valid
# kinematic parent tree.  Keep it for provenance, but never infer payload
# descendants from it.
PWIF3D_MODEL_GRAPH_EDGES = [
    (0, 1), (0, 2), (2, 5), (3, 0), (4, 2), (5, 7), (6, 3),
    (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 9), (13, 11),
]

# Person-in-WiFi-3D joint order, as reported row-by-row in Table 3 of the
# official CVPR 2024 paper.  This is *not* CrowdPose order even though the
# upstream repository also contains generic CrowdPose visualization code.
PWIF3D_JOINT_NAMES = (
    'neck', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'left_hip', 'right_elbow', 'right_hip',
    'left_hand', 'left_knee', 'right_hand', 'right_knee',
    'left_ankle', 'right_ankle',
)

# Explicit rooted kinematic tree for payload descendants.  DT-Pose's model
# graph above contains the additional (5, 7) left-hip/right-hip relation and
# therefore has one cycle.  That lateral relation is useful to a GCN but is not
# a parent-child bone; excluding it gives the exact acyclic tree implicitly
# selected by the legacy BFS and used to produce the manuscript's pivot-3,
# pivot-5 and pivot-7 geometry rows.
PWIF3D_KINEMATIC_EDGES = [
    (0, 1),
    (0, 2), (2, 4), (4, 8), (2, 5), (5, 9), (9, 12),
    (0, 3), (3, 6), (6, 10), (3, 7), (7, 11), (11, 13),
]
PWIF3D_EDGES = PWIF3D_KINEMATIC_EDGES  # backwards-compatible public name

N_JOINTS = 14
ROOT = 0  # neck; pivot=3 rotates the right-side branch {6,7,10,11,13}

def _build_tree(edges=PWIF3D_EDGES, n=N_JOINTS, root=ROOT):
    if len(edges) != n - 1:
        raise ValueError(
            f'kinematic graph must contain n-1 edges, got {len(edges)} for n={n}')
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        if a not in adj or b not in adj or a == b:
            raise ValueError(f'invalid kinematic edge {(a, b)}')
        adj[a].append(b); adj[b].append(a)
    parent = {root: None}; order = [root]; seen = {root}
    qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v] = u; order.append(v)
    if len(seen) != n:
        raise ValueError(
            f'kinematic graph is disconnected: reached {len(seen)} of {n} joints')
    children = {i: [] for i in range(n)}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    return parent, children, adj


PARENT, CHILDREN, ADJ = _build_tree()


def descendants(pivot):
    out = []
    stack = list(CHILDREN[pivot])
    while stack:
        j = stack.pop(); out.append(j); stack.extend(CHILDREN[j])
    return sorted(out)


def terminal_chains():
    leaves = [j for j in range(N_JOINTS) if not CHILDREN[j]]
    chains = []
    for leaf in leaves:
        path = [leaf]; p = PARENT[leaf]
        while p is not None:
            path.append(p); p = PARENT[p]
        path = path[::-1]                              # root ... leaf
        # report the distal 3-joint segment (pivot, mid, distal) when available
        seg = path[-3:] if len(path) >= 3 else path
        chains.append({'leaf': leaf, 'root_path': path,
                       'pivot': seg[0], 'segment': seg,
                       'rotated_joints': descendants(seg[0])})
    return chains

# --------------------------------------------------------------------------- rotation
def _rodrigues(axis, theta):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        raise ValueError('rotation axis must be non-zero')
    axis = axis / norm
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def g_dose(dose, theta_max=np.deg2rad(60.0), mode='linear'):
    """Monotone dose -> rotation-angle map. dose in [0,1]."""
    dose = float(np.clip(dose, 0.0, 1.0))
    if mode == 'linear':
        return theta_max * dose
    if mode == 'sqrt':
        return theta_max * np.sqrt(dose)
    if mode == 'quad':
        return theta_max * dose ** 2
    raise ValueError(mode)


def rotate_subchain(pose, pivot, theta, axis=(0.0, 0.0, 1.0)):
    pose = np.array(pose, float, copy=True)
    rot_js = descendants(pivot)
    if not rot_js:
        return pose
    R = _rodrigues(axis, theta)
    pj = pose[..., pivot, :]                            # (...,3) pivot position
    for j in rot_js:
        rel = pose[..., j, :] - pj
        pose[..., j, :] = (rel @ R.T) + pj
    return pose


def make_target_pose(pose, pivot, dose, theta_max=np.deg2rad(60.0),
                     mode='linear', axis=(0.0, 0.0, 1.0)):
    return rotate_subchain(pose, pivot, g_dose(dose, theta_max, mode), axis)


def subchain_bone_lengths(pose, pivot):
    js = [pivot] + descendants(pivot)
    L = []
    for j in js:
        p = PARENT[j]
        if p is not None and p in js:
            L.append(np.linalg.norm(pose[..., j, :] - pose[..., p, :], axis=-1))
    return np.array(L)


def all_bone_lengths(pose):
    """Lengths of every edge in the verified payload kinematic tree."""
    pose = np.asarray(pose)
    return np.stack([
        np.linalg.norm(pose[..., a, :] - pose[..., b, :], axis=-1)
        for a, b in PWIF3D_KINEMATIC_EDGES
    ], axis=-1)
