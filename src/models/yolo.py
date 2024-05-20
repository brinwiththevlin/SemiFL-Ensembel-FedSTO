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

    def forward():
        raise NotImplementedError


class BackBone(nn.Module):
    def __init__(self, w: int, d: int, r: int) -> None:
        """YOLOv8 backbone

        Args:
            w (int): width multiplier
            d (int): depth multiplier
            r (int): resolution multiplier
        """
        super().__init__()
        self.conv1 = Conv(3, int(64 * w), 3, 2, 1)
        self.conv2 = Conv(int(64 * w), int(128 * w), 3, 2, 1)
        self.C2f1 = C2F(int(128 * w), int(128 * w), int(3 * d), True)
        self.conv3 = Conv(int(128 * w), int(256 * w), 3, 2, 1)
        self.C2f2 = C2F(int(256 * w), int(256 * w), int(6 * d), True)
        self.conv4 = Conv(int(256 * w), int(512 * w), 3, 2, 1)
        self.C2f3 = C2F(int(512 * w), int(512 * w), int(6 * d), True)
        self.conv5 = Conv(int(512 * w), int(512 * w * r), 3, 2, 1)
        self.C2f4 = C2F(int(512 * w * r), int(512 * w * r), int(6 * d), True)
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.C2f1(x)
        x = self.conv3(x)
        x = self.C2f2(x)
        residual1 = x
        x = self.conv4(x)
        x = self.C2f3(x)
        residual2 = x
        x = self.conv5(x)
        x = self.C2f4(x)
        x = self.sppf(x)
        return residual1, residual2, x


class Neck(nn.Module):
    def __init__(self, c2f_2: torch.Tensor, c2f_3: torch.Tensor, sppf: torch.Tensor, w: int, d: int, r: int) -> None:
        """model neck, takes intermediate outputs of backbone and sppf

        Args:
            c2f_2 (torch.Tensor): output of second c2f block. shape: (80, 80, 256*w)
            c2f_3 (torch.Tensor): output of third c2f block. shape: (40, 40, 512*w)
            sppf (torch.Tensor): output of sppf block. shape: (20, 20, 512*w*r)
            w (int): width multiplier
            d (int): depth multiplier
            r (int): resolution multiplier
        """
        super().__init__()
        self.bb_residual1 = c2f_2
        self.bb_residual2 = c2f_3
        self.bb_output = sppf
        self.up1 = nn.Upsample(scale_factor=2)
        self.C2f1 = C2F(int(512 * w * (r + 1)), int(512 * w), int(3 * d), False)
        self.up2 = nn.Upsample(scale_factor=2)
        self.C2f2 = C2F(int(768 * w), int(256 * w), int(3 * d), False)
        self.conv1 = Conv(int(256 * w), int(256 * w), 3, 2, 1)
        self.c2f3 = C2F(int(768 * w), int(512 * w), int(3 * d), False)
        self.conv2 = Conv(int(512 * w), int(512 * w), 3, 2, 1)
        self.c2f4 = C2F(int(512 * w * (1 + r)), int(512 * w * r), int(3 * d), False)

    def forward(self):
        x = self.up1(self.bb_output)
        x = torch.cat([x, self.bb_residual2], dim=1)
        x = self.C2f1(x)
        c2f1 = x
        x = self.up2(x)
        x = torch.cat([x, self.bb_residual1], dim=1)
        x = self.C2f2(x)
        residual1 = x
        x = self.conv1(x)
        x = torch.cat([x, c2f1], dim=1)
        x = self.c2f3(x)
        residual2 = x
        x = self.conv2(x)
        x = torch.cat([x, self.bb_output], dim=1)
        x = self.c2f4(x)
        return residual1, residual2, x


class Head(nn.Module):
    def __init__(self, c2f_2: torch.Tensor, c2f_3: torch.Tensor, c2f_4: torch.Tensor, w: int, d: int, r: int) -> None:
        """head of the model

        Args:
            c2f_2 (torch.Tensor): output of second c2f block. shape: (80, 80, 256*w)
            c2f_3 (torch.Tensor): output of third c2f block. shape: (40, 40, 512*w)
            c2f_4 (torch.Tensor): output of fourth c2f block. shape: (20, 20, 512*w*r)
            w (int): width multiplier
            d (int): depth multiplier
            r (int): resolution multiplier
        """
        self.residual1 = c2f_2
        self.residual2 = c2f_3
        self.residual3 = c2f_4
        self.detect1 = Detect()  # TODO: Implement Detect class
        self.detect2 = Detect()  # TODO: Implement Detect class
        self.detect3 = Detect()  # TODO: Implement Detect class

    def forward(self):
        x = self.detect1(self.residual1)
        y = self.detect2(self.residual2)
        z = self.detect3(self.residual3)
        return x, y, z


class YOLOv8(nn.Module):
    def __init__(self, mode):
        assert mode in ["n", "s", "m", "l", "x"], f"Invalid mode {mode}"
        self.w = w_vaules[mode]
        self.d = d_vaules[mode]
        self.r = r_vaules[mode]
        self.backbone = BackBone(self.w, self.d, self.r)
        self.neck = Neck(self.w, self.d, self.r)
        self.head = Head(self.w, self.d, self.r)

    def forward(self, x: torch.Tensor):
        residual1, residual2, bb_output = self.backbone(x)
        residual1, residual2, neck_output = self.neck(residual1, residual2, bb_output)
        x, y, z = self.head(residual1, residual2, neck_output)
        return x, y, z
