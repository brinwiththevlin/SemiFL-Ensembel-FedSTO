import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn
from config import cfg

d_vaules: dict = {"n": 0.333, "s": 0.333, "m": 0.667, "l": 1.0, "x": 1.0}
w_vaules: dict = {"n": 0.25, "s": 0.5, "m": 0.75, "l": 1.0, "x": 1.25}
r_vaules: dict = {"n": 2.0, "s": 2.0, "m": 1.5, "l": 1.0, "x": 1.0}


class Conv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = self.bn(out)
        out = self.silu(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.shorcut = shortcut
        self.conv1 = Conv(channels, channels, 3, 1, 1)
        self.conv2 = Conv(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shorcut:
            out += x
        return out


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1, 1, 0)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = Conv(in_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        x = torch.cat([x, pool1, pool2, pool3], dim=1)
        x = self.conv2(x)
        return x


class C2F(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_bottelneck: int, shortcut: bool = True) -> None:
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1, 1, 0)
        self.bottleneck = nn.Sequential(*[BottleNeck(out_channels // 2, shortcut) for _ in range(num_bottelneck)])
        self.conv2 = Conv(0.5 * (num_bottelneck + 2) * out_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        y1, y2 = x.split(x.size(1) // 2, dim=1)
        outputs = [y1, y2]
        for layer in self.bottleneck:
            y2 = layer(y2)
            outputs.append(y2)
        x = torch.cat(outputs, dim=1)
        x = self.conv2(x)
        return x


class Detect(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        raise NotImplementedError


class BackBone(nn.Module):
    def __init__(self, mode: chr) -> None:
        assert mode in ["n", "s", "m", "l", "x"], f"Invalid mode {mode}"
        super().__init__()
        self.mode = mode
        d = d_vaules[mode]
        w = w_vaules[mode]
        r = r_vaules[mode]
        self.conv1 = Conv(3, int(64 * w), 3, 2, 1)
        self.conv2 = Conv(int(64 * w), int(128 * w), 3, 2, 1)
        self.C2f1 = C2F(int(128 * w), int(128 * w), int(3 * d), True)
        self.conv3 = Conv(int(128 * w), int(256 * w), 3, 2, 1)
        self.C2f2 = C2F(int(256 * w), int(256 * w), int(6 * d), True)
        self.conv4 = Conv(int(256 * w), int(512 * w), 3, 2, 1)
        self.C2f3 = C2F(int(512 * w), int(512 * w), int(6 * d), True)
        self.conv5 = Conv(int(512 * w), int(512 * w * r), 3, 2, 1)
        self.C2f4 = C2F(int(512 * w * r), int(512 * w * r), int(6 * d), True)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.C2f1(x)
        x = self.conv3(x)
        x = self.C2f2(x)
        x = self.conv4(x)
        x = self.C2f3(x)
        x = self.conv5(x)
        x = self.C2f4(x)
        return x
