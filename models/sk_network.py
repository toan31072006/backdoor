import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, input_dim, output_dim, dim1, dim2, pool_dim,
                 M=4, G=1, r=4, stride=1, L=32):
        super().__init__()
        self.dim1, self.dim2, self.output_dim = dim1, dim2, output_dim
        self.M, self.pool_dim = M, pool_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 3, stride, padding=1 + i,
                          dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True))
            for i in range(M)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if pool_dim == 'freq-chan':
            d = int(output_dim / r)
            self.fc = nn.Sequential(nn.Conv1d(output_dim, d, 1, 1),
                                    nn.BatchNorm1d(d), nn.ReLU(inplace=True))
            self.fcs = nn.ModuleList([nn.Conv1d(d, output_dim, 1, 1) for _ in range(M)])
        else:
            raise NotImplementedError(pool_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b = x.shape[0]
        feats = torch.cat([c(x) for c in self.convs], dim=1)
        feats = feats.view(b, self.M, feats.shape[2], self.output_dim, feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        # (b, dim1, C) -> (b, C, dim1): self.fc is a 1x1 Conv1d over the channel
        # axis, so this must be a real transpose.  A .view() with swapped dims
        # keeps the shape but scrambles the content (clean MPJPE 94 -> ~140 mm).
        feats_S = torch.mean(feats_U, dim=3).permute(0, 2, 1)
        feats_Z = self.fc(feats_S)
        av = torch.cat([fc(feats_Z) for fc in self.fcs], dim=1)
        av = av.view(b, self.M, self.output_dim, self.dim1, 1)
        av = self.softmax(av).view(b, self.M, self.dim1, self.output_dim, 1)
        feats_V = torch.sum(feats * av, dim=1)
        return torch.transpose(feats_V, 1, 2)


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, dim1, dim2,
                 pool_dim, M=4, G=1, r=4, stride=1, L=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_features), nn.ReLU(inplace=True))
        self.conv2_sk = nn.Sequential(
            SKConv(mid_features, out_features, dim1, dim2, pool_dim,
                   M=4, G=1, r=4, stride=1, L=32),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv2_sk(self.conv1(x))
