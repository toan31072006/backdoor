import copy, math
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm

class Channel_Embeddings(nn.Module):
    def __init__(self, img_size, in_channels):
        super().__init__()
        n_patches = img_size[0] * img_size[1]
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, num_keypoints):
        super().__init__()
        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.num_keypoints = num_keypoints

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = self.num_keypoints, 12
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return self.activation(self.norm(self.conv(x)))

class Attention_org(nn.Module):
    def __init__(self, channel_num, num_heads):
        super().__init__()
        self.KV_size = channel_num
        self.num_heads = num_heads
        self.query1 = nn.ModuleList([nn.Linear(channel_num, channel_num, False) for _ in range(num_heads)])
        self.key = nn.ModuleList([nn.Linear(channel_num, channel_num, False) for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(channel_num, channel_num, False) for _ in range(num_heads)])
        self.psi = nn.InstanceNorm2d(num_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num, channel_num, False)
        self.attn_dropout = Dropout(0.1); self.proj_dropout = Dropout(0.1)

    def forward(self, emb1):
        Q = torch.stack([q(emb1) for q in self.query1], 1).transpose(-1, -2)
        K = torch.stack([k(emb1) for k in self.key], 1)
        V = torch.stack([v(emb1) for v in self.value], 1)
        scores = torch.matmul(Q, K) / math.sqrt(self.KV_size)
        probs = self.attn_dropout(self.softmax(self.psi(scores)))
        ctx = torch.matmul(probs, V.transpose(-1, -2)).permute(0, 3, 2, 1).mean(3)
        return self.proj_dropout(self.out1(ctx)), None


class Mlp(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.fc1 = nn.Linear(c, hidden); self.fc2 = nn.Linear(hidden, c)
        self.act = nn.GELU(); self.drop = Dropout(0.1)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block_ViT(nn.Module):
    def __init__(self, channel_num, num_heads):
        super().__init__()
        self.attn_norm1 = LayerNorm(channel_num, eps=1e-6)
        self.channel_attn = Attention_org(channel_num, num_heads)
        self.ffn_norm1 = LayerNorm(channel_num, eps=1e-6)
        self.ffn1 = Mlp(channel_num, channel_num * 4)

    def forward(self, emb1):
        cx, _ = self.channel_attn(self.attn_norm1(emb1))
        emb1 = emb1 + cx
        return emb1 + self.ffn1(self.ffn_norm1(emb1)), None


class Encoder(nn.Module):
    def __init__(self, channel_num, num_layers, num_heads):
        super().__init__()
        self.layer = nn.ModuleList([copy.deepcopy(Block_ViT(channel_num, num_heads))
                                    for _ in range(num_layers)])
        self.encoder_norm1 = LayerNorm(channel_num, eps=1e-6)

    def forward(self, emb1):
        for blk in self.layer:
            emb1, _ = blk(emb1)
        return self.encoder_norm1(emb1), None


class ChannelTransformer(nn.Module):
    def __init__(self, img_size, channel_num, num_layers, num_heads, num_keypoints):
        super().__init__()
        self.embeddings_1 = Channel_Embeddings(img_size, channel_num)
        self.encoder = Encoder(channel_num, num_layers, num_heads)
        self.reconstruct_1 = Reconstruct(channel_num, channel_num, 1, (1, 1), num_keypoints)

    def forward(self, en1):
        emb1 = self.embeddings_1(en1)
        enc1, _ = self.encoder(emb1)
        return self.reconstruct_1(enc1) + en1, None
