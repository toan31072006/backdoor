import torchvision
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from models.channel_trans import ChannelTransformer


class MetaFiNet(nn.Module):
    def __init__(self, num_keypoints=14, num_coor=3, num_person=1,
                 dataset='person-in-wifi-3d', pretrained=False):
        super().__init__()
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        self.diff = num_keypoints * num_person - 17
        rn = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder_conv1_p1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.encoder_bn1_p1 = rn.bn1
        self.encoder_relu_p1 = rn.relu
        self.encoder_layer1_p1 = rn.layer1
        self.encoder_layer2_p1 = rn.layer2
        self.encoder_layer3_p1 = rn.layer3
        self.encoder_layer4_p1 = rn.layer4
        self.tf = ChannelTransformer([num_keypoints * num_person, 12], 512, 1, 3,
                                     num_keypoints * num_person)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_coor, 1, 1, 0, bias=False), nn.BatchNorm2d(num_coor), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm1d(num_coor)
        self.bn2 = nn.BatchNorm2d(512)

    def _enc(self, xi):
        xi = self.encoder_relu_p1(self.encoder_bn1_p1(self.encoder_conv1_p1(xi)))
        xi = self.encoder_layer1_p1(xi); xi = self.encoder_layer2_p1(xi)
        xi = self.encoder_layer3_p1(xi); xi = self.encoder_layer4_p1(xi)
        return xi

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1)                                # (b,1,3,180,20)
        H = 136 + 8 * self.diff
        rsz = Resize([H, 32])
        outs = []
        for c in range(3):
            xc = x[:, :, c:c + 1, :, :]
            xc = torch.flatten(torch.transpose(xc, 2, 3), 3, 4)
            outs.append(self._enc(rsz(xc)))
        x = self.bn2(torch.cat(outs, dim=3))
        x, _ = self.tf(x)
        fea = x.mean(3).mean(2)
        x = self.decode(x)
        x = nn.AvgPool2d((1, 12), stride=(1, 1))(x).squeeze(dim=3)
        x = self.bn1(x)
        x = torch.transpose(x, 1, 2).view(b, self.num_person, self.num_keypoints, self.num_coor)
        return x, fea


def metafi_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
