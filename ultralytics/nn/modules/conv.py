# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from typing import List
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "FastERBlock",
    "NAM",
    "FasterNetBlock",
)

class NAM(nn.Module):
    def __init__(self, channels, reduction=2):
        super(NAM, self).__init__()
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

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa

        return x

class FastERBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        out_channels = max(8, out_channels)  # Мінімум 8 каналів
        mid_channels = max(1, out_channels // expansion)  # Мінімум 1 канал
        groups = mid_channels if mid_channels > 1 else 1  # Мінімальне значення groups = 1

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x + identity if identity.shape == x.shape else x



# class FastERNet(nn.Module):
#     """FastERNet як бекбоун для YOLO."""
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True)
#         )
#         self.stage1 = FastERBlock(32, 64)
#         self.stage2 = FastERBlock(64, 128)
#         self.stage3 = FastERBlock(128, 256)
#         self.stage4 = FastERBlock(256, 512)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, 1000)  # У YOLO ця частина буде видалена
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         return x

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class FasterNetBlock(nn.Module):
    """Complete FasterNet Block using MLP and Partial Convolutions"""
    def __init__(self, dim, n_div=4, mlp_ratio=4.0, drop_path=0.1, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.mlp_block = MLPBlock(
            dim=dim,
            n_div=n_div,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.ReLU, inplace=True),
            pconv_fw_type='split_cat',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp_block(self.norm(x))

class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return outs


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


# class ChannelAttention(nn.Module):
#     """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
#
#     def __init__(self, channels: int) -> None:
#         """Initializes the class and sets the basic configurations and instance variables required."""
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
#         return x * self.act(self.fc(self.pool(x)))
#
#
# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""
#
#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         # assert kernel_size in (3, 7), "kernel size must be 3 or 7"
#         kernel_size = 7
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
#
#
# class CBAM(nn.Module):
#     """Convolutional Block Attention Module."""
#
#     def __init__(self, c1, kernel_size=7):
#         """Initialize CBAM with given input channel (c1) and kernel size."""
#         super().__init__()
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         """Applies the forward pass through C1 module."""
#         return self.spatial_attention(self.channel_attention(x))

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, scale=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Пулінг зменшує розмір до (batch, channels, 1, 1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
        # self.act = nn.Hardsigmoid()
        self.scale = scale  # Контроль сили впливу CBAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.act(self.fc(self.pool(x)))
        # return self.scale * x * attention

        return x + self.scale * x * attention # fixed 4


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, scale=0.5):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        # kernel_size = 7
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
        # self.act = nn.Hardsigmoid()
        self.scale = scale

    def forward(self, x):
        return x + self.scale * x * self.act(self.cv1(torch.cat([
            torch.mean(x, 1, keepdim=True),  # Середнє значення по каналах
            torch.max(x, 1, keepdim=True)[0]  # Максимум по каналах
        ], 1)))
        # return self.scale * x * self.act(self.cv1(torch.cat([
        #     torch.mean(x, 1, keepdim=True),  # Середнє значення по каналах
        #     torch.max(x, 1, keepdim=True)[0]  # Максимум по каналах
        # ], 1)))

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7, attention_scale=1.0, channel_scale=1.0):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, scale=channel_scale)
        self.spatial_attention = SpatialAttention(kernel_size, scale=attention_scale)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
