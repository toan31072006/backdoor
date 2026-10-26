"""SKConv must transpose feats_S with a real permute, not a shape-only view.

self.fc is a 1x1 Conv1d over the channel axis, so feats_S must be laid out as
(b, C, dim1).  A .view(b, C, dim1) on a (b, dim1, C) tensor has the right shape
but scrambled content; it silently degraded clean MPJPE from ~94 to ~140 mm.
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.sk_network import SKConv


def _unit():
    return SKConv(3, 8, dim1=16, dim2=4, pool_dim='freq-chan',
                  M=1, G=1, r=4, stride=1, L=32)


def test_fc_is_conv1d_over_channels():
    u = _unit()
    fc0 = u.fc[0]
    assert isinstance(fc0, torch.nn.Conv1d)
    assert fc0.in_channels == u.output_dim


def test_feats_S_is_transposed_not_viewed():
    torch.manual_seed(0)
    u = _unit().eval()
    x = torch.randn(2, 3, 16, 4)
    captured = {}
    def _grab(m, i, o):
        captured['in'] = i[0]      # return None so the hook does not replace fc's output
    h = u.fc.register_forward_hook(_grab)
    with torch.no_grad():
        u(x)
    h.remove()
    feats = torch.cat([c(x) for c in u.convs], 1)
    feats = feats.view(2, u.M, feats.shape[2], u.output_dim, feats.shape[3])
    S = torch.mean(torch.sum(feats, 1), 3)              # (b, dim1, C)
    assert torch.allclose(captured['in'], S.permute(0, 2, 1), atol=1e-6)
    assert not torch.equal(captured['in'],
                           S.contiguous().view(2, S.shape[2], S.shape[1]))
