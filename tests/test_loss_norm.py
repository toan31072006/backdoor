"""Contracts for the paper's ordinary-ERM victim objective.

The victim is allowed to see only an ordinary ``(csi, pose)`` supervised
batch. Poison membership, dose, pivot, and target-joint identities must not
enter its objective. These tests replace the former decomposed-loss tests,
which encoded a stronger attacker than the paper's threat model permits.
"""

import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_backdoor import (  # noqa: E402
    _mpjpe_loss,
    _resolve_training_config,
    _validate_training_contract,
    _victim_update,
)


class _ToyVictim(torch.nn.Module):
    def __init__(self, n_joints=4):
        super().__init__()
        self.head = torch.nn.Linear(3, n_joints * 3, bias=False)
        self.n_joints = n_joints

    def forward(self, csi):
        pooled = csi.mean(dim=(2, 3))
        pose = self.head(pooled).reshape(-1, 1, self.n_joints, 3)
        return pose, None


def test_mpjpe_matches_paper_equation_over_full_batch():
    generator = torch.Generator().manual_seed(7)
    pred = torch.randn(5, 1, 14, 3, generator=generator)
    target = torch.randn(5, 1, 14, 3, generator=generator)

    expected = torch.linalg.vector_norm(pred - target, dim=-1).mean()
    actual = _mpjpe_loss(pred, target)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_mpjpe_does_not_upweight_a_small_joint_or_sample_group():
    pred = torch.zeros(2, 1, 4, 3)
    target = torch.zeros_like(pred)
    pred[1, 0, 3, 0] = 8.0

    # One joint with error 8 among 2 * 1 * 4 equally weighted joints.
    assert float(_mpjpe_loss(pred, target)) == pytest.approx(1.0)


def test_mpjpe_rejects_mismatched_pose_shapes():
    with pytest.raises(ValueError, match='matching pose shapes'):
        _mpjpe_loss(torch.zeros(2, 1, 14, 3), torch.zeros(2, 14, 3))


def test_victim_update_api_cannot_accept_attack_metadata():
    assert tuple(inspect.signature(_victim_update).parameters) == (
        'model', 'optimizer', 'csi', 'target')


def test_victim_update_is_an_ordinary_mpjpe_step():
    torch.manual_seed(4)
    model = _ToyVictim()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    csi = torch.randn(3, 3, 6, 5)
    target = torch.randn(3, 1, 4, 3)

    with torch.no_grad():
        expected = float(_mpjpe_loss(model(csi)[0], target))
    before = [p.detach().clone() for p in model.parameters()]
    actual = _victim_update(model, optimizer, csi, target)

    assert actual == pytest.approx(expected)
    assert any(not torch.equal(old, new) for old, new in
               zip(before, model.parameters()))


@pytest.mark.parametrize('forbidden_key', [
    'lambda_target', 'lambda_nontarget', 'loss_norm',
])
def test_paper_contract_rejects_attack_specific_victim_options(forbidden_key):
    with pytest.raises(ValueError, match='attack-specific'):
        _validate_training_contract({'victim_loss': 'mpjpe', forbidden_key: 1})


def test_paper_contract_accepts_only_standard_mpjpe():
    _validate_training_contract({})
    _validate_training_contract({'victim_loss': 'mpjpe'})
    with pytest.raises(ValueError, match="victim_loss must be 'mpjpe'"):
        _validate_training_contract({'victim_loss': 'oracle_decomposed'})


def test_resolved_config_records_the_exact_piw3d_paper_recipe():
    cfg = _resolve_training_config({})
    assert cfg['seed'] == 42
    assert cfg['victim_loss'] == 'mpjpe'
    assert cfg['trigger'] == 'micro_dropper'
    assert cfg['poison_select'] == 'uniform'
    assert cfg['dose_grid'] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert cfg['optimizer'] == 'adamw'
    assert cfg['weight_decay'] == pytest.approx(0.01)
    assert cfg['num_person'] == 1

    with pytest.raises(ValueError, match='requires optimizer=adamw'):
        _resolve_training_config({'optimizer': 'sgd'})

    tsba = _resolve_training_config({'trigger': 'tsba'})
    assert tsba['tsba_eps'] == pytest.approx(0.1)
    assert tsba['tsba_hidden'] == 32
    assert tsba['tsba_lr'] == pytest.approx(1e-3)
    assert tsba['tsba_warmup_epochs'] == 10
    assert tsba['tsba_generator_steps'] == 4
