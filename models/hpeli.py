import torch
import torch.nn as nn
from models.sk_network import SKUnit


class HPELiNet(nn.Module):
    def __init__(self, num_keypoints=14, num_coor=3, subcarrier_num=180,
                 num_person=1, dataset='person-in-wifi-3d'):
        super().__init__()
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        num_lay = 64
        self.skunit1 = SKUnit(3, num_lay, num_lay, dim1=subcarrier_num, dim2=10,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(num_lay, num_lay * 2, num_lay * 2,
                              dim1=subcarrier_num // 2, dim2=8,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        # regression head (person-in-wifi-3d geometry): 128x45x5 -> 640 -> K*C*P
        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(32, 16, (3, 1), (1, 1), 0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(640, num_keypoints * num_coor * num_person))

    def forward(self, x):
        b = x.shape[0]
        m = nn.AvgPool2d((2, 2))
        x = m(self.skunit1(x))
        out1 = m(self.skunit2(x))
        fea = out1.mean(3).mean(2)
        x = self.regression(out1)
        x = x.reshape(b, self.num_person, self.num_keypoints, self.num_coor)
        return x, fea


def hpeli_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
