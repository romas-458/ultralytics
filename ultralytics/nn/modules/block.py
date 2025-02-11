# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2f_Light",
    "C2f_Light2",
    "C2f_CBAM",
    "C2f_CBAM_Light",
    "C2f_SE",
    "C2f_EMA",
    "C2f_CA",
    "C2f_NAM",
    "C2f_ECA",
    "C2f_GAM",
    "C2f_SA",
    "C2f_CBAM_Res",
    "C2f_ICBAM",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_Light(nn.Module):
    """Легкий C2f без уваги для максимальної швидкості."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.375):  # 🔥 Менше каналів (e=0.375)
        super().__init__()
        self.c = int(c2 * e)  # Менше прихованих каналів
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + int(n * 0.75)) * self.c, c2, 1)  # 🔥 Менше Bottleneck-блоків
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=0.75) for _ in
                               range(int(n * 0.75)))  # 🔥 Менше шарів
        self.act = nn.LeakyReLU(0.1, inplace=True)  # 🔥 Швидша активація

    def forward(self, x):
        """Forward pass через легкий C2f без уваги."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return out  # ✅ Без уваги (максимальна швидкість)

class C2f_Light2(nn.Module):
    """Легкий C2f без уваги для максимальної швидкості."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.375):  # 🔥 Менше каналів (e=0.375)
        super().__init__()
        self.c = int(c2 * e)  # Менше прихованих каналів
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + int(n * 1)) * self.c, c2, 1)  # 🔥 Менше Bottleneck-блоків
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass через легкий C2f без уваги."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return out  # ✅ Без уваги (максимальна швидкість)

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, kernel_size=7, scale=1.0):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.scale = scale

    def forward(self, x):
        ca = self.channel_attention(x)  # Канальна увага
        sa = self.spatial_attention(torch.cat([torch.mean(x, dim=1, keepdim=True),
                                               torch.max(x, dim=1, keepdim=True)[0]], dim=1))  # Просторова увага
        return x * (1 + self.scale * ca) * (1 + self.scale * sa)  # Пом'якшена версія CBAM


class C2f_CBAM(nn.Module):
    """C2f + CBAM: Faster Implementation of CSP Bottleneck with CBAM."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, cbam_scale=1.0):
        """Initialize CSP bottleneck with CBAM."""
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cbam = CBAM(c2, scale=cbam_scale)  # Додаємо CBAM

    def forward(self, x):
        """Forward pass через C2f + CBAM."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення вхідного тензора
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        return self.cbam(self.cv2(torch.cat(y, 1)))  # Об'єднання та CBAM

    def forward_split(self, x):
        """Forward pass using split() замість chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))  # CBAM після об'єднання

class C2f_CBAM_Light(nn.Module):
    """Легкий C2f з оригінальним CBAM"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.375, cbam_scale=1.0):  # 🔥 Менше каналів (e=0.375)
        super().__init__()
        self.c = int(c2 * e)  # Менше прихованих каналів
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + int(n * 0.75)) * self.c, c2, 1)  # 🔥 Менше Bottleneck-блоків
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=0.75) for _ in range(int(n * 0.75)))  # 🔥 Менше шарів
        self.cbam = CBAM(c2, scale=cbam_scale)  # ✅ Використовуємо оригінальний CBAM

    def forward(self, x):
        """Forward pass через C2f + CBAM."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.cbam(out)  # ✅ CBAM після вихідного Conv


# ✅ Improved Channel Attention (ICBAM)
class IChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

# ✅ Improved Spatial Attention (ICBAM)
class ISpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=(3, 5)):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_out = self.conv1(torch.cat([avg_out, max_out], dim=1)) + self.conv2(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(spatial_out)

# ✅ ICBAM Module (Improved CBAM)
class ICBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = IChannelAttention(channels, reduction)
        self.spatial_attention = ISpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class C2f_ICBAM(nn.Module):
    """C2f з удосконаленим ICBAM"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.icbam = ICBAM(c2)  # 🔥 Використовуємо ICBAM

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.icbam(out)  # ✅ Удосконалений ICBAM


# ✅ ResBlock для CBAM
class ResBlock(nn.Module):
    """ResNet-блок для CBAM, покращує передачу градієнтів"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x  # 🔥 Skip connection
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + res)  # 🔥 Додаємо skip connection

class CBAM_Res(nn.Module):
    """CBAM з ResBlock для кращої передачі градієнтів"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.resblock = ResBlock(channels)  # 🔥 Додаємо ResBlock

    def forward(self, x):
        res = x  # 🔥 Skip connection
        # ✅ Channel Attention
        x = x * self.channel_attention(x)
        # ✅ ResBlock для покращення передачі градієнтів
        x = self.resblock(x)
        # ✅ Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = x * self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x + res  # 🔥 Skip connection

class C2f_CBAM_Res(nn.Module):
    """C2f з CBAM + ResBlock"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # Приховані канали
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cbam_res = CBAM_Res(c2)  # 🔥 Використовуємо оновлений CBAM

    def forward(self, x):
        """Forward pass через C2f з CBAM + ResBlock"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.cbam_res(out)  # ✅ CBAM з ResBlock


# ✅ SE Attention Module
class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention Module (SE Attention)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # GAP (Squeeze)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # Ваги для каналів
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_pool(x).view(b, c)  # (B, C, 1, 1) → (B, C)
        y = self.fc(y).view(b, c, 1, 1)  # (B, C) → (B, C, 1, 1)
        return x * y  # Помножуємо канали на ваги

# ✅ C2f (CSP Bottleneck з SE Attention)
class C2f_SE(nn.Module):
    """C2f + SE: CSP Bottleneck з двома конволюціями та SE Attention."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 1x1 Conv (Final)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.se = SEAttention(c2, reduction)  # ✅ Додаємо SE Attention

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        return self.se(self.cv2(torch.cat(y, 1)))  # ✅ SE Attention після об'єднання

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.se(self.cv2(torch.cat(y, 1)))  # ✅ SE Attention після об'єднання


# ✅ Виправлена версія Coordinate Attention (CA)
class CAAttention(nn.Module):
    """Coordinate Attention (CA) для YOLOv8."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))  # Усереднення по висоті
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))  # Усереднення по ширині

        hidden_dim = max(8, channels // reduction)  # Мінімальна кількість каналів - 8
        self.conv1 = nn.Conv2d(2 * channels, hidden_dim, kernel_size=1, bias=False)  # ✅ Виправлено
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)  # ✅ Виправлено
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # ✅ Усереднення по висоті та ширині
        y_h = self.h_pool(x)  # (B, C, H, 1)
        y_w = self.w_pool(x).permute(0, 1, 3, 2)  # (B, C, 1, W) → (B, C, W, 1)

        # ✅ Вирівнюємо до однакової форми
        y_h = y_h.expand(-1, -1, h, w)  # (B, C, H, W)
        y_w = y_w.expand(-1, -1, h, w)  # (B, C, H, W)

        # Об'єднуємо дві ознаки
        y = torch.cat([y_h, y_w], dim=1)  # (B, 2C, H, W)

        # Пропускаємо через Conv + ReLU + Sigmoid
        y = self.relu(self.conv1(y))  # ✅ Вхід = 2C каналів, вихід = hidden_dim
        y = self.conv2(y).sigmoid()  # ✅ Вихід = (B, C, H, W)

        return x * y  # ✅ Коректне помноження


class C2f_CA(nn.Module):
    """C2f + Coordinate Attention (CA) для YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний 1x1 Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ca = CAAttention(c2, reduction)  # ✅ Додаємо Coordinate Attention

    def forward(self, x):
        """Forward pass через C2f + CA Attention."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.ca(out)  # ✅ CA Attention застосовується тільки після вихідного Conv


# ✅ Efficient Multi-scale Attention (EMA)
class EMAAttention(nn.Module):
    """Efficient Multi-scale Attention (EMA) для YOLOv8."""

    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Глобальне усереднення
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # ✅ Глобальний контекст
        global_context = self.global_pool(x)  # (B, C, 1, 1)
        global_context = self.conv1(global_context)  # (B, C//r, 1, 1)
        global_context = self.conv2(global_context)  # (B, C, 1, 1)

        # ✅ Локальні особливості
        local_context = self.depthwise_conv(x)  # (B, C, H, W)
        local_context = self.bn(local_context)

        # ✅ Об'єднуємо глобальні та локальні особливості
        attention_map = self.sigmoid(global_context + local_context)

        return x * attention_map  # Застосовуємо увагу до вихідного тензора


# ✅ C2f + EMA (Efficient Multi-scale Attention)
class C2f_EMA(nn.Module):
    """C2f + EMA Attention для YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16, kernel_size=3):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний 1x1 Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ema = EMAAttention(c2, reduction, kernel_size)  # ✅ Додаємо EMA Attention

    def forward(self, x):
        """Forward pass через C2f + EMA Attention."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.ema(out)  # ✅ EMA Attention застосовується тільки після вихідного Conv


# ✅ NAM (Norm-Aware Attention Module)
class NAMAttention(nn.Module):
    """Norm-Aware Attention Module (NAM) для YOLOv8."""

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True)  # ✅ Додає нормалізацію
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)  # Просторова увага
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ✅ Нормалізація через BN
        norm_x = self.bn(x)

        # ✅ Просторова увага (обчислюється на середніх та максимальних значеннях)
        spatial_map = torch.mean(norm_x, dim=1, keepdim=True) + torch.max(norm_x, dim=1, keepdim=True)[0]
        attention_map = self.sigmoid(self.conv(spatial_map))

        return x * attention_map  # ✅ Коригуємо вихідний тензор

class C2f_NAM(nn.Module):
    """C2f + NAM Attention для YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=7):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний 1x1 Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.nam = NAMAttention(c2, kernel_size)  # ✅ Додаємо NAM Attention

    def forward(self, x):
        """Forward pass через C2f + NAM Attention."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.nam(out)  # ✅ NAM Attention після вихідного Conv

class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA)"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        attention = self.conv(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(attention)

class C2f_ECA(nn.Module):
    """C2f + ECA Attention"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.eca = ECAAttention(c2, kernel_size)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.eca(out)

# ✅ GAM Attention (Global Attention Mechanism)
class GAMAttention(nn.Module):
    """Global Attention Mechanism (GAM) для YOLOv8."""

    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        # Channel Attention
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_ca = nn.Sigmoid()

        # Spatial Attention
        self.conv_sa = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_sa = nn.Sigmoid()

    def forward(self, x):
        # ✅ Channel Attention
        avg_out = self.global_avg(x)
        max_out = self.global_max(x)
        ca = self.fc2(self.relu(self.fc1(avg_out + max_out)))  # Об'єднуємо GAP та GMP
        ca = self.sigmoid_ca(ca)

        # ✅ Spatial Attention
        mean_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.conv_sa(torch.cat([mean_x, max_x], dim=1))
        sa = self.sigmoid_sa(sa)

        return x * ca * sa  # Об'єднання обох механізмів уваги

class C2f_GAM(nn.Module):
    """C2f + GAM Attention для YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16, kernel_size=3):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний 1x1 Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.gam = GAMAttention(c2, reduction, kernel_size)  # ✅ Додаємо GAM Attention

    def forward(self, x):
        """Forward pass через C2f + GAM Attention."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.gam(out)  # ✅ GAM Attention після вихідного Conv


from torch.nn.parameter import Parameter

# ✅ Оновлена версія Shuffle Attention (SA)
class ShuffleAttention(nn.Module):
    """Shuffle Attention (SA) для YOLOv8."""

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        """Перестановка каналів для покращення узгодженості ознак."""
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)  # Розбивка на групи
        x = x.permute(0, 2, 1, 3, 4)  # Перемішування груп
        x = x.reshape(b, -1, h, w)  # Відновлення вихідної форми
        return x

    def forward(self, x):
        b, c, h, w = x.size()

        # ✅ Розбиваємо канали на G груп (B*G, C//G, H, W)
        x = x.view(b * self.G, -1, h, w)  # (B*G, C/G, H, W)

        # ✅ Поділ каналів на дві частини
        x_0, x_1 = x.chunk(2, dim=1)  # (B*G, C//(2*G), H, W) x2

        # ✅ Channel Attention
        x_channel = self.avg_pool(x_0)  # (B*G, C//(2*G), 1, 1)
        x_channel = self.cweight * x_channel + self.cbias  # CA корекція
        x_channel = x_0 * self.sigmoid(x_channel)

        # ✅ Spatial Attention
        x_spatial = self.gn(x_1)  # Групова нормалізація
        x_spatial = self.sweight * x_spatial + self.sbias  # SA корекція
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # ✅ Об'єднання ознак
        out = torch.cat([x_channel, x_spatial], dim=1)  # (B*G, C/G, H, W)
        out = out.contiguous().view(b, -1, h, w)  # (B, C, H, W)

        # ✅ Перемішування каналів (Channel Shuffle)
        out = self.channel_shuffle(out, 2)
        return out

class C2f_SA(nn.Module):
    """C2f + Shuffle Attention для YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, groups=8):
        super().__init__()
        self.c = int(c2 * e)  # Приховані канали
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Вхідний 1x1 Conv
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Вихідний 1x1 Conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.sa = ShuffleAttention(c2, G=groups)  # ✅ Додаємо Shuffle Attention

    def forward(self, x):
        """Forward pass через C2f + Shuffle Attention."""
        y = list(self.cv1(x).chunk(2, 1))  # Розділення каналів
        y.extend(m(y[-1]) for m in self.m)  # Проходження через Bottleneck
        out = self.cv2(torch.cat(y, 1))  # Об'єднання ознак через Conv
        return self.sa(out)  # ✅ SA Attention після вихідного Conv



class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
