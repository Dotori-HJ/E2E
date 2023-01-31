import warnings
from functools import lru_cache, partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from .init import constant_init, kaiming_init, xavier_init


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_layer=nn.Conv1d,
        norm_layer=nn.SyncBatchNorm,
        act=partial(nn.ReLU, inplace=True),
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_layer is not None
        self.with_activation = act is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels


            try:
                self.norm = norm_layer(norm_channels)
            except TypeError:
                self.norm = norm_layer(num_channels=norm_channels)
            # self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            self.activate = act()

        # Use msra init by default
        self.init_weights()

    # @property
    # def norm(self):
    #     return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            kaiming_init(self.conv, a=0, nonlinearity='relu')
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                if isinstance(self.norm, nn.LayerNorm):
                    ndim = x.ndim
                    axis_order = list(range(ndim))
                    axis_order = [0] + axis_order[2:] + [1]
                    x = x.permute(axis_order)
                x = self.norm(x)
                if isinstance(self.norm, nn.LayerNorm):  # convert to channel first
                    ndim = x.ndim
                    axis_order = [0, ndim - 1] + list(range(1, ndim - 1))
                    x = x.permute(axis_order)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x



class TDM(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(
        self,
        in_channels,
        stage_layers=(1, 1, 1, 1),
        kernel_sizes=3,
        strides=2,
        paddings=1,
        dilations=1,
        out_channels=256,
        conv_layer=nn.Conv1d,
        norm_layer=nn.SyncBatchNorm,
        act=partial(nn.ReLU, inplace=True),
        out_indices=(0, 1, 2, 3, 4),
    ):
        super(TDM, self).__init__()

        self.in_channels = in_channels
        self.num_stages = len(stage_layers)
        self.stage_layers = stage_layers
        self.kernel_sizes = _ntuple(self.num_stages)(kernel_sizes)
        self.strides = _ntuple(self.num_stages)(strides)
        self.paddings = _ntuple(self.num_stages)(paddings)
        self.dilations = _ntuple(self.num_stages)(dilations)
        self.out_channels = _ntuple(self.num_stages)(out_channels)
        self.out_indices = out_indices

        assert (
            len(self.stage_layers)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.paddings)
            == len(self.dilations)
            == len(self.out_channels)
        )

        self.td_layers = []
        for i in range(self.num_stages):
            td_layer = self.make_td_layer(
                self.stage_layers[i],
                in_channels,
                self.out_channels[i],
                self.kernel_sizes[i],
                self.strides[i],
                self.paddings[i],
                self.dilations[i],
                conv_layer,
                norm_layer,
                act,
            )
            in_channels = self.out_channels[i]
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, td_layer)
            self.td_layers.append(layer_name)

    @staticmethod
    def make_td_layer(
        num_layer,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        conv_layer,
        norm_layer,
        act,
    ):
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                act=act,
            )
        )
        for _ in range(1, num_layer):
            layers.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_layer=conv_layer,
                    norm_layer=norm_layer,
                    act=act,
                )
            )

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initiate the parameters."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)


class FPN(nn.Module):
    """Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        conv_layer=nn.Conv1d,
        lateral_norm_layer=nn.SyncBatchNorm,
        fpn_norm_layer=partial(nn.GroupNorm, num_groups=32),
        act=None,
        upsample_cfg=dict(mode="nearest"),
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_layer=conv_layer,
                norm_layer=lateral_norm_layer,
                act=act,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_layer=conv_layer,
                norm_layer=fpn_norm_layer,
                act=act,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        for proj in self.fpn_convs:
            nn.init.xavier_uniform_(proj.weight, gain=1)
            nn.init.zeros_(proj.bias)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                # This is a workaround when converting PyTorch model
                # to ONNX model
                prev_shape = tuple(map(lambda x: int(x), prev_shape))
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        # build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)
