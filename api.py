import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adadelta, SGD
import tqdm


class AutoEncodeDecode32(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.channels = channels

        self.c1 = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5),  # bx1x32x32 -> bx16x28x28
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=2),  # -> bx64x12x12
            nn.ReLU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=2),  # -> bx256x4x4
            nn.ReLU(),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1),  # bx256x1x1
            nn.Tanh(),
        )

        self.tc1 = nn.Sequential(
            nn.ConvTranspose2d(256, 1024, kernel_size=4),  # bx1024x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=1),
            nn.ReLU(),
        )

        self.tc2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=5, dilation=2),  # bx128x12x12
            nn.ReLU(),
        )

        self.tc3 = nn.Sequential(
            nn.ConvTranspose2d(256, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5),  # bx1x32x32
            nn.Tanh(),
        )

    def decode(self, input):
        x1 = self.c1(input)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.tc1(x3)
        x5 = self.tc2(x4)
        x6 = self.tc3(x5)
        return [input, x1, x2, x3, x4, x5, x6]

    def forward(self, input):
        x1 = self.c1(input)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.tc1(x3)
        x5 = self.tc2(x4)
        x6 = self.tc3(x5)
        return x6

