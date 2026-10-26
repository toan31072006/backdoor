import pathlib
import sys
import unittest

import numpy as np
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack.payload import (
    N_JOINTS,
    PARENT,
    PWIF3D_JOINT_NAMES,
    PWIF3D_KINEMATIC_EDGES,
    PWIF3D_MODEL_GRAPH_EDGES,
    ROOT as KINEMATIC_ROOT,
    _build_tree,
    all_bone_lengths,
    descendants,
    make_target_pose,
)
from attack.poison import PoisonedDataset, collate
from attack.tsba import TSBATrigger
from eval.metrics import attack_metrics
from models.hpeli import HPELiNet
from train_backdoor import _generator_update, _validate_training_contract


def _pose_batch(n_samples=4, n_joints=14):
    joint = np.arange(n_joints, dtype=np.float32)
    pose = np.stack((joint * 0.10,
                     (joint * 7 % 3) * 0.08,
                     (joint * 5 % 4) * 0.06), axis=-1)
    return np.repeat(pose[None], n_samples, axis=0)


class _DuplicatePoseBase:
    def __init__(self, n):
        self.items = [{'kpt': str(i)} for i in range(n)]

    @staticmethod
    def load_pose(_):
        return np.zeros((1, 14, 3), dtype=np.float32)


class _ToyDatasetBase:
    """In-memory fixed-trigger dataset with the production feeder interface."""

    def __init__(self, n):
        self.items = [
            {'csi': i, 'kpt': i, 'name': f'sample-{i:03d}'} for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_raw(index):
        return np.full((3, 180, 20), float(index + 1), dtype=np.float32)

    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32)

    @staticmethod
    def load_pose(index):
        pose = _pose_batch(1)[0] + np.float32(index * 1e-4)
        return pose[None]


class _FixedTrigger:
    requires_deferred_injection = False

    @staticmethod
    def inject(raw, dose, eps=0.3):
        return np.asarray(raw, dtype=np.float32) + np.float32(dose * eps)


class _ToyPoseVictim(torch.nn.Module):
    def __init__(self, n_joints=14):
        super().__init__()
        self.n_joints = n_joints
        self.head = torch.nn.Linear(3, n_joints * 3)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        return self.head(pooled).reshape(-1, 1, self.n_joints, 3), None


class RegressionTests(unittest.TestCase):
    def test_tsba_shape_dose_and_multiplicative_bound(self):
        trigger = TSBATrigger(n_ant=3, hidden=8).eval()
        x = torch.rand(2, 3, 180, 20) + 0.05
        with torch.no_grad():
            zero = trigger.inject_tensor(x, 0.0, eps=0.1)
            full = trigger.inject_tensor(x, torch.ones(2), eps=0.1)
        torch.testing.assert_close(zero, x, rtol=0.0, atol=0.0)
        relative = (full / x - 1.0).abs()
        self.assertLessEqual(float(relative.max()), 0.100001)
        self.assertEqual(tuple(full.shape), tuple(x.shape))

    def test_tsba_update_changes_generator_not_frozen_victim(self):
        victim = _ToyPoseVictim()
        trigger = TSBATrigger(hidden=8)
        optimizer = torch.optim.Adam(trigger.parameters(), lr=1e-3)
        victim_before = [p.detach().clone() for p in victim.parameters()]
        trigger_before = [p.detach().clone() for p in trigger.parameters()]
        loss = _generator_update(
            victim, trigger, optimizer, torch.rand(3, 3, 12, 5),
            torch.rand(3, 1, 14, 3), torch.ones(3),
            {'tsba_eps': 0.1, 'pivot': 7})
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(trigger_before, trigger.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(victim_before, victim.parameters())))
        self.assertTrue(all(p.requires_grad for p in victim.parameters()))

    def test_diverse_fps_never_repeats_identical_poses(self):
        selector = PoisonedDataset.__new__(PoisonedDataset)
        selector.base = _DuplicatePoseBase(10)
        idx = selector._select_poison(
            np.random.default_rng(42), n=10, n_pois=4, select='diverse')
        self.assertEqual(len(idx), 4)
        self.assertEqual(len(set(idx.tolist())), 4)

    def test_poison_count_uses_floor_and_plan_is_reproducible(self):
        base = _ToyDatasetBase(11)
        kwargs = dict(mode='train', rho=0.25, dose_min=0.2, dose_max=1.0,
                      pivot=3, theta_max_deg=90.0, select='uniform')
        first = PoisonedDataset(base, _FixedTrigger(), seed=73, **kwargs)
        replay = PoisonedDataset(base, _FixedTrigger(), seed=73, **kwargs)

        # floor(0.25 * 11) == 2, whereas the old round() implementation gave 3.
        self.assertEqual(first.n_poison, 2)
        self.assertEqual(first.poison_plan, replay.poison_plan)
        self.assertEqual(first.manifest(), replay.manifest())
        self.assertEqual(len(set(first.poison_idx)), first.n_poison)
        self.assertTrue(all(0.2 <= dose <= 1.0
                            for _, dose in first.poison_plan))

    def test_fixed_trigger_training_batch_contains_only_csi_and_pose(self):
        ds = PoisonedDataset(
            _ToyDatasetBase(8), _FixedTrigger(), mode='train', rho=0.5,
            dose_min=0.2, dose_max=1.0, pivot=3, theta_max_deg=90.0,
            seed=7, select='uniform')

        samples = [ds[i] for i in range(4)]
        self.assertTrue(all(set(sample) == {'csi', 'pose'}
                            for sample in samples))
        self.assertEqual(set(collate(samples)), {'csi', 'pose'})

    def test_payload_uses_an_explicit_acyclic_kinematic_tree(self):
        self.assertEqual(len(PWIF3D_KINEMATIC_EDGES), N_JOINTS - 1)
        self.assertEqual(PARENT[KINEMATIC_ROOT], None)
        self.assertEqual(KINEMATIC_ROOT, 0)
        self.assertEqual(PWIF3D_JOINT_NAMES[KINEMATIC_ROOT], 'neck')
        parent_edges = {
            frozenset((joint, parent))
            for joint, parent in PARENT.items() if parent is not None
        }
        expected_edges = {frozenset(edge) for edge in PWIF3D_KINEMATIC_EDGES}
        self.assertEqual(parent_edges, expected_edges)
        self.assertEqual(set(PARENT), set(range(N_JOINTS)))

        # The upstream GCN adjacency contains a cycle and must never be silently
        # converted into the payload's parent tree.  The kinematic tree is the
        # DT-Pose graph with only its lateral hip relation removed.
        self.assertNotEqual(PWIF3D_MODEL_GRAPH_EDGES,
                            PWIF3D_KINEMATIC_EDGES)
        model_edges = {frozenset(edge) for edge in PWIF3D_MODEL_GRAPH_EDGES}
        kinematic_edges = {
            frozenset(edge) for edge in PWIF3D_KINEMATIC_EDGES
        }
        self.assertEqual(model_edges - kinematic_edges, {frozenset((5, 7))})
        self.assertEqual(kinematic_edges, model_edges - {frozenset((5, 7))})
        with self.assertRaises(ValueError):
            _build_tree(PWIF3D_MODEL_GRAPH_EDGES, N_JOINTS, KINEMATIC_ROOT)

    def test_official_joint_order_and_manuscript_pivot_semantics(self):
        self.assertEqual(PWIF3D_JOINT_NAMES, (
            'neck', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'left_hip', 'right_elbow', 'right_hip',
            'left_hand', 'left_knee', 'right_hand', 'right_knee',
            'left_ankle', 'right_ankle',
        ))

        expected = {
            3: ('right_shoulder',
                [6, 7, 10, 11, 13],
                ['right_elbow', 'right_hip', 'right_hand',
                 'right_knee', 'right_ankle']),
            5: ('left_hip',
                [9, 12],
                ['left_knee', 'left_ankle']),
            7: ('right_hip',
                [11, 13],
                ['right_knee', 'right_ankle']),
        }
        for pivot, (pivot_name, joint_ids, joint_names) in expected.items():
            self.assertEqual(PWIF3D_JOINT_NAMES[pivot], pivot_name)
            self.assertEqual(descendants(pivot), joint_ids)
            self.assertEqual([PWIF3D_JOINT_NAMES[j] for j in joint_ids],
                             joint_names)

    def test_canonical_main_and_clean_configs_use_right_hip_subtree(self):
        config_dir = ROOT / 'configs'
        canonical = (
            'attack.yaml',
            'control_rho0_lr1e-3.yaml',
            'piw_p7_th90_uniform.yaml',
            'piw_p7_th90_uniform_seed0.yaml',
            'piw_p7_th90_uniform_seed1.yaml',
            'piw_p7_th90_tsba_unif.yaml',
        )
        for name in canonical:
            with self.subTest(config=name):
                cfg = yaml.safe_load(
                    (config_dir / name).read_text(encoding='utf-8'))
                self.assertEqual(cfg['pivot'], 7)
                self.assertEqual(PWIF3D_JOINT_NAMES[cfg['pivot']], 'right_hip')
                self.assertEqual(descendants(cfg['pivot']), [11, 13])

    def test_every_pivot7_config_uses_ordinary_mpjpe_contract(self):
        paths = sorted((ROOT / 'configs').glob('piw_p7_th90*.yaml'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                _validate_training_contract(cfg)
                self.assertEqual(cfg['victim_loss'], 'mpjpe')
                self.assertFalse({'lambda_target', 'lambda_nontarget',
                                  'loss_norm'} & set(cfg))

    def test_every_kinematic_bone_is_preserved_for_every_valid_pivot(self):
        pose = np.random.default_rng(19).normal(size=(5, N_JOINTS, 3))
        original_lengths = all_bone_lengths(pose)
        for pivot in range(N_JOINTS):
            rotated = descendants(pivot)
            if not rotated:
                continue
            target = make_target_pose(
                pose, pivot=pivot, dose=0.73, theta_max=np.deg2rad(87.0),
                axis=(1.0, 2.0, 3.0))
            np.testing.assert_allclose(
                all_bone_lengths(target), original_lengths,
                rtol=1e-10, atol=1e-10, err_msg=f'pivot={pivot}')
            untouched = [j for j in range(N_JOINTS) if j not in rotated]
            np.testing.assert_allclose(target[:, untouched], pose[:, untouched],
                                       rtol=0.0, atol=0.0)

    def test_direction_metrics_use_endpoint_vector_from_paper(self):
        true = _pose_batch()
        pivot = 3
        target = make_target_pose(
            true, pivot=pivot, dose=1.0, theta_max=np.deg2rad(90.0))
        js = descendants(pivot)
        pred_clean = true.copy()
        pred_clean[:, js] += np.array([0.12, -0.05, 0.08], np.float32)
        pred_attack = pred_clean.copy()
        pred_attack[:, js] += 0.5 * (target[:, js] - pred_clean[:, js])

        got = attack_metrics(pred_attack, target, pred_clean, true, pivot)
        u = (pred_attack[:, js] - pred_clean[:, js]).reshape(len(true), -1)
        v = (target[:, js] - pred_clean[:, js]).reshape(len(true), -1)
        q = (target[:, js] - true[:, js]).reshape(len(true), -1)
        expected_gain = ((u * v).sum(-1) / ((v * v).sum(-1) + 1e-12)).mean()
        expected_cos = ((u * v).sum(-1) /
                        (np.linalg.norm(u, axis=-1) *
                         np.linalg.norm(v, axis=-1) + 1e-12)).mean()
        payload_gain = ((u * q).sum(-1) / ((q * q).sum(-1) + 1e-12)).mean()

        self.assertAlmostEqual(got['effect_gain_mean'], expected_gain, places=6)
        self.assertAlmostEqual(got['cosine_alignment_mean'], expected_cos,
                               places=6)
        self.assertAlmostEqual(got['effect_gain_mean'], 0.5, places=6)
        self.assertNotAlmostEqual(got['effect_gain_mean'], payload_gain,
                                  places=4)

    def test_asr_rejects_unchanged_and_zero_dose(self):
        true = _pose_batch()
        pivot = 3
        target = make_target_pose(
            true, pivot=pivot, dose=1.0, theta_max=np.deg2rad(90.0))
        unchanged = attack_metrics(true, target, true, true, pivot=pivot)
        self.assertEqual(unchanged['asr'], 0.0)
        self.assertAlmostEqual(unchanged['target_residual_ratio_mean'], 1.0,
                               places=5)

        perfect = attack_metrics(target, target, true, true, pivot=pivot)
        self.assertEqual(perfect['asr'], 1.0)

        # Endpoint direction remains exact when the clean prediction is biased
        # on the attacked joint, provided the triggered output lands at G(Y,d).
        pred_clean = true.copy()
        pred_clean[:, descendants(pivot)] += np.array(
            [0.05, -0.02, 0.03], dtype=np.float32)
        pred_attack = target.copy()
        biased = attack_metrics(pred_attack, target, pred_clean, true,
                                pivot=pivot)
        self.assertEqual(biased['asr'], 1.0)
        self.assertAlmostEqual(biased['cosine_alignment_mean'], 1.0, places=6)
        self.assertAlmostEqual(biased['effect_gain_mean'], 1.0, places=6)

        zero = attack_metrics(true, true, true, true, pivot=pivot)
        self.assertEqual(zero['asr'], 0.0)

    def test_hpeli_person_forward_shape(self):
        model = HPELiNet(subcarrier_num=180, dataset='person-in-wifi-3d')
        model.eval()
        with torch.no_grad():
            pred, _ = model(torch.randn(2, 3, 180, 20))
        self.assertEqual(tuple(pred.shape), (2, 1, 14, 3))


if __name__ == '__main__':
    unittest.main()
