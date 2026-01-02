
# reference: https://medium.com/@AIchemizt/attention-u-net-in-pytorch-step-by-step-guide-with-code-and-explanation-417d80a6dfd0

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttenttionGate(nn.Module):
    def __init__(self, g_c, s_c, o_c):
        super().__init__()
        self.wg = nn.Sequential(
            nn.Conv2d(g_c, o_c, kernel_size=1),
            nn.BatchNorm2d(o_c)
        )

        self.ws = nn.Sequential(
            nn.Conv2d(s_c, o_c, kernel_size=1),
            nn.BatchNorm2d(o_c)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(o_c, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.wg(g)
        s1 = self.ws(s)
        out = F.relu(g1 + s1)
        psi = self.psi(out)
        return s * psi
    
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        self.attention = AttenttionGate(in_c, skip_c, out_c)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
    
    def forward(self, x):
        return x