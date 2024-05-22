import torch
import torch.nn as nn
from .utils import init_param, loss_fn
from typing import Tuple


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, in_channels: int, num_classes: int, reg_max: int) -> None:
        self.conv1 = Conv(in_channels, in_channels, 3, 1, 1)
        self.conv2 = Conv(in_channels, in_channels, 3, 1, 1)
        self.conv3 = Conv(in_channels, in_channels, 3, 1, 1)
        self.conv4 = Conv(in_channels, in_channels, 3, 1, 1)
        self.reg_max = reg_max
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = nn.Conv2d(x1, 4 * self.reg_max, 1, 1, 0)
        x2 = self.conv3(x)
        x2 = self.conv4(x2)
        x2 = nn.Conv2d(x2, self.num_classes, 1, 1, 0)
        return x1, x2


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, w: int, d: int, r: int) -> None:
        """model neck, takes intermediate outputs of backbone and sppf

        Args:
            w (int): width multiplier
            d (int): depth multiplier
            r (int): resolution multiplier
        """
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2)
        self.C2f1 = C2F(int(512 * w * (r + 1)), int(512 * w), int(3 * d), False)
        self.up2 = nn.Upsample(scale_factor=2)
        self.C2f2 = C2F(int(768 * w), int(256 * w), int(3 * d), False)
        self.conv1 = Conv(int(256 * w), int(256 * w), 3, 2, 1)
        self.c2f3 = C2F(int(768 * w), int(512 * w), int(3 * d), False)
        self.conv2 = Conv(int(512 * w), int(512 * w), 3, 2, 1)
        self.c2f4 = C2F(int(512 * w * (1 + r)), int(512 * w * r), int(3 * d), False)

    def forward(
        self, c2f_2: torch.Tensor, c2f_3: torch.Tensor, sppf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.bb_residual1 = c2f_2
        self.bb_residual2 = c2f_3
        self.bb_output = sppf
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
    def __init__(self, w: int, d: int, r: int, num_classes: int, reg_max: int) -> None:
        """head of the model

        Args:
            w (int): width multiplier
            d (int): depth multiplier
            r (int): resolution multiplier
            num_classes (int): number of classes
            reg_max (int): maximum number of regions
        """
        super().__init__()
        self.detect1 = Detect(in_channels=256 * w, num_classes=num_classes, reg_max=reg_max)
        self.detect2 = Detect(in_channels=512 * w, num_classes=num_classes, reg_max=reg_max)
        self.detect3 = Detect(in_channels=512 * w * r, num_classes=num_classes, reg_max=reg_max)

    def forward(
        self, c2f_2: torch.Tensor, c2f_3: torch.Tensor, c2f_4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.residual1 = c2f_2
        self.residual2 = c2f_3
        self.residual3 = c2f_4
        x = self.detect1(self.residual1)
        y = self.detect2(self.residual2)
        z = self.detect3(self.residual3)
        return x, y, z


class YOLOv8(nn.Module):
    def __init__(
        self,
        mode,
        num_classes: int,
        reg_max: int,
        pre_train: bool = False,
        freeze: bool = False,
        bb_wights: dict = None,
    ) -> None:
        """YOLOv8 model

        Args:
            mode (char): model size. one of ["n", "s", "m", "l", "x"]
            num_classes (int): number of classes
            reg_max (int): maximum number of regions
            pre_train (bool, optional): if pre_train is true, only train the backbone. other parts are frozen. Defaults to False.
            freeze(bool, optional): if freeze is true, only train the backbone and neck. head is frozen. Defaults to False.
            bb_wights (dict, optional): path to backbone weights. Defaults to None.
        """
        assert mode in ["n", "s", "m", "l", "x"], f"Invalid mode {mode}"
        assert num_classes > 0, "num_classes should be greater than 0"
        assert reg_max > 0, "reg_max should be greater than 0"
        assert (pre_train and bb_wights) or not pre_train, "pre_train should be true only if bb_weights are provided"
        super().__init__()
        self.w = w_vaules[mode]
        self.d = d_vaules[mode]
        self.r = r_vaules[mode]
        self.backbone = BackBone(self.w, self.d, self.r)
        self.neck = Neck(self.w, self.d, self.r)
        self.head = Head(self.w, self.d, self.r, num_classes, reg_max)
        self.pre_train = pre_train
        self.freeze = freeze

        if not pre_train:
            init_param(self.backbone)
            init_param(self.neck)
            init_param(self.head)
        else:
            self.backbone.load_state_dict(bb_wights)

        if freeze:
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual1, residual2, bb_output = self.backbone(x)
        residual1, residual2, neck_output = self.neck(residual1, residual2, bb_output)
        x, y, z = self.head(residual1, residual2, neck_output)
        return x, y, z
