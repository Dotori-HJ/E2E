"""
Video encoder modules.
"""
import logging
import math
from collections import OrderedDict
from typing import Dict, List

import ipdb as pdb
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn.modules.normalization import GroupNorm
from torchvision.models._utils import IntermediateLayerGetter

from models.video_encoder_archs.slowfast import ResNet3dSlowFast
from models.video_encoder_archs.tsm import TSM
from models.video_encoder_archs.video_mae import VisionTransformer
from models.video_encoder_archs.video_swin import SwinTransformer3D, params
from opts import cfg
from util.misc import NestedTensor, is_main_process


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

def unfold(ip, kernel_size, stride):
    '''Expect NCTHW shaped tensor, extract sliding block for snippet-wise feature extraction'''
    # ip_ncts = rearrange(ip_ncthw, "n c t h w -> n c t (h w)")
    # ip_ncts = F.unfold(ip_ncts, (kernel_size, 1), stride=(stride, 1), padding=((kernel_size-stride)//2, 1))
    N, C, T, H, W = ip.shape
    pad_size = (( kernel_size - stride ) // 2, (kernel_size-stride+1) // 2)
    ip_pad = F.pad(ip, (0, 0, 0, 0, *pad_size), mode='constant', value=0)
    num_windows = T // stride
    start = torch.arange(num_windows).reshape([num_windows, 1]) * stride
    indices = (start + torch.arange(kernel_size)).view(-1)  # (num_windows, kernel_size)
    out = torch.index_select(ip_pad, dim=2, index=indices.to(ip.device))
    # pdb.set_trace()
    out= out.reshape(N, C, num_windows, kernel_size, H, W)
    out = rearrange(out, 'n c nw ks h w -> (n nw) c ks h w')
    return out

class IdentityNeck(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        # return (x[-1], )
        # return [x for x in features[self.indices]]
        return [x[i] for i in self.indices]


class TunerBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels

        # Conv
        # self.norm = LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        self._init_weight()

    def _init_weight(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv1.weight)
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class Tuner(nn.Module):
    def __init__(self, base_channels, num_lvls, middle_channels=2048):
        super().__init__()
        self.base_channels = base_channels
        self.num_lvls = num_lvls

        self.layers = nn.ModuleList([
            TunerBlock(
                int(base_channels * 2 ** lvl),
                middle_channels,
                int(base_channels * 2 ** (lvl + 1)),
            )
            for lvl in range(num_lvls)
        ])

    def forward(self, features):
        assert len(features) == self.num_lvls + 1, f"feature_len={len(features)}, num_levels={self.num_lvls}"

        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(features[i]) + features[i+1]
            # elif i == self.num_lvls - 1:
            #     out = layer(out)
            else:
                out = layer(out) + features[i+1]

        return out

# class PyramidTuner(nn.Module):
#     def __init__(self, feature_dims:tuple, middle_dim, output_dim):
#         super().__init__()
#         kernel_size = 1
#         self.proj_layers = nn.ModuleList([
#             nn.Conv1d(dim, middle_dim, kernel_size=kernel_size, padding=kernel_size//2)
#             for dim in feature_dims
#         ])
#         self.middle_layers = nn.ModuleList([
#             TunerBlock(middle_dim, 2048, middle_dim, kernel_size=kernel_size)
#             # nn.Conv1d(middle_dim, middle_dim, kernel_size=3)
#             for _ in range(len(feature_dims))
#         ])
#         self.output_layer = nn.Conv1d(middle_dim, output_dim, kernel_size=kernel_size, padding=kernel_size//2)

#         # self.scaler = nn.Parameter(torch.ones(1))
#         self._init_weights()
#         # self.apply(self._init_weights)

#     def _init_weights(self):
#         with torch.no_grad():
#             for layer in self.proj_layers:
#                 nn.init.kaiming_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)

#             nn.init.zeros_(self.output_layer.weight)
#             nn.init.zeros_(self.output_layer.bias)

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Conv1d):
#     #         # nn.init.kaiming_normal_(m, nn.Conv1d)
#     #         trunc_normal_(m.weight, std=.02)
#     #         if isinstance(m, nn.Conv1d) and m.bias is not None:
#     #             nn.init.zeros_(m.bias)
#     #     elif isinstance(m, nn.LayerNorm):
#     #         nn.init.ones_(m.weight)
#     #         nn.init.zeros_(m.bias)

#     def forward(self, features):
#         proj_features = [layer(x) for x, layer in zip(features, self.proj_layers)]

#         for i, layer in enumerate(self.middle_layers):
#             if i == 0:
#                 out = layer(proj_features[i]) + proj_features[i+1]
#             elif i != len(self.middle_layers) - 1:
#                 out = layer(out) + proj_features[i+1]
#             else:
#                 out = layer(out)
#         # out = self.output_layer(out) * self.scaler + features[-1]
#         out = self.output_layer(out) + features[-1]

#         return out


class PyramidTuner(nn.Module):
    def __init__(self, feature_dims:tuple, middle_dim, output_dim):
        super().__init__()
        kernel_size = 1
        self.proj_layers = nn.ModuleList([
            nn.Conv1d(dim, middle_dim, kernel_size=kernel_size, padding=kernel_size//2)
            for dim in feature_dims
        ])
        self.middle_layers = nn.ModuleList([
            TunerBlock(middle_dim, 2048, middle_dim, kernel_size=3)
            # nn.Conv1d(middle_dim, middle_dim, kernel_size=3)
            for _ in range(len(feature_dims))
        ])
        # self.output_layer = nn.Conv1d(middle_dim, output_dim, kernel_size=kernel_size, padding=kernel_size//2)

        # self._init_weights()

    def forward(self, features):
        proj_features = [layer(x) for x, layer in zip(features, self.proj_layers)]

        outs = []
        for i, layer in enumerate(self.middle_layers):
            if i == 0:
                out = layer(proj_features[-i])
            elif i != len(self.middle_layers) - 1:
                out = layer(out) + proj_features[-i+1]
            else:
                out = layer(out)
            outs.append(out)
        # out = self.output_layer(out) * self.scaler + features[-1]
        # out = self.output_layer(out) + features[-1]

        return outs


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, conv=False, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        if conv:
            self.linear1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
            self.linear2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        else:
            self.linear1 = nn.Linear(in_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, out_dim)

        # if in_dim != out_dim:
        #     if conv:
        #         self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        #     else:
        #         self.proj = nn.Linear(in_dim, out_dim)
        # else:
        #     self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.linear1.bias)
            # nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
            # if hasattr(self.proj, 'bias'):
            #     nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x))) + x

class Pooler(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1,
            padding=pool_size//2,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x) - x

class Mixer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, temporal_length, conv=False):
        super().__init__()
        self.conv = conv
        self.norm1 = LayerNorm(in_dim)
        self.mixer = MLP(temporal_length, int(temporal_length * 4), temporal_length)
        # self.mixer = Pooler()
        self.norm2 = LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim, out_dim, conv=conv)

        # self.channel_mlp = MLP(temporal_length, hidden_dim, temporal_length)

    def forward(self, x):
        if self.conv:
            x = self.mixer(self.norm1(x))
            x = self.mlp(self.norm2(x))
        else:
            x = self.mixer(self.norm1(x))
            x = self.mlp(self.norm2(x).transpose(2, 1)).transpose(2, 1)
        return x

class MixerTuner(nn.Module):
    def __init__(self, feature_dims, middle_dim, temporal_length, num_layers=1):
        super().__init__()
        self.feature_dims = feature_dims
        self.middle_dim = middle_dim
        self.num_layers = num_layers

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, self.middle_dim, kernel_size=1),
                nn.GroupNorm(32, self.middle_dim)
            )
            for feature_dim in feature_dims
        ])
        self.mixers = nn.ModuleList([
            Mixer(
                int(4 * middle_dim),
                int(4 * middle_dim * 2),
                int(4 * middle_dim),
                temporal_length,
                conv=True,
            ) for i in range(num_layers)
        ])
        self.norm = LayerNorm(int(4 * middle_dim))

    def forward(self, features):
        features = [layer(x) for layer, x in zip(self.input_projs, features)]
        multi_features = torch.cat(features, dim=1)

        for layer in self.mixers:
            multi_features = layer(multi_features)
        multi_features = self.norm(multi_features)

        return [multi_features]

# class AttentionMixer(nn.Module):
#     def __init__(self, feature_dims, middle_dim, kernel_size=1):
#         self.num_levels = len(feature_dims)

#         self.proj_layers = nn.ModuleList([
#             nn.Conv1d(dim, middle_dim, kernel_size=kernel_size, padding=kernel_size//2)
#             for dim in feature_dims
#         ])
#         self.level_emb = nn.Embedding(self.num_levels + 1, middle_dim)
#         self.emb_token = nn.Embedding(self.)
#         self.norm
#         self.

class SimpleMixer(nn.Module):
    def __init__(self, feature_dims, middle_dim):
        super().__init__()
        self.linear1 = nn.Linear(feature_dims[-1], middle_dim)
        self.linear2 = nn.Linear(middle_dim, feature_dims[-1])

        self._init_weight()

    def _init_weight(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, features):
        x = features[-1]
        return self.linear2(F.relu(self.linear1(x.transpose(2, 1)))).transpose(2, 1) + x


class VideoEncoder(nn.Module):
    def __init__(self, arch='slowfast', fix_encoder=False, neck='pyramid'):
        super().__init__()
        self.arch = arch
        self.use_upsample = cfg.temporal_upsample

        if arch == 'slowfast':
            self.backbone = ResNet3dSlowFast(None, depth=cfg.slowfast_depth,freeze_bn=cfg.freeze_bn, freeze_bn_affine=cfg.freeze_affine, slow_upsample=cfg.slow_upsample)
            self.num_channels = 2304
            self.pyramid_channels = (288, 576, 1152, 2304)
            self.base_channels = 512
            temporal_length = 64

        elif arch in ['tsm', 'tsn']:
            self.backbone = TSM(arch=cfg.tsm_base_model, is_shift=arch=='tsm')
            self.num_channels = self.backbone.out_channels
        elif arch == 'video_mae':
            self.backbone = VisionTransformer()
            self.num_channels = self.backbone.num_channels
        elif arch == 'video_swin':
            self.backbone = SwinTransformer3D(pretrained=cfg.pretrained_model, **params[cfg.size])
            self.num_channels = self.backbone.num_features[-1]
            self.pyramid_channels = self.backbone.num_features
            self.base_channels = 512
            temporal_length = 128
        else:
            raise ValueError('Not supported arch: {}'.format(arch))

        indices = [-1]
        self.fix_encoder = fix_encoder

        if fix_encoder:
            self._fix_encoder()

        if neck == 'identity':
            self.neck = IdentityNeck(indices)
        elif neck == 'pyramid':
            self.neck = PyramidTuner(self.pyramid_channels, self.base_channels, self.num_channels)
            self.pyramid_channels = [self.base_channels for _ in self.pyramid_channels]
        elif neck == "tuner":
            self.neck = Tuner(288, 2304, 3)
        elif neck == "mixer":
            self.neck = MixerTuner(self.pyramid_channels, 256, temporal_length)
            self.pyramid_channels = [int(256 * 4) for _ in self.pyramid_channels]
        elif neck == "simple_mixer":
            self.neck = SimpleMixer(self.pyramid_channels, self.base_channels)
        else:
            assert True, f"neck={neck}"

        self.pyramid_channels = [self.pyramid_channels[i] for i in indices]


    def forward(self, tensor_list):
        '''tensor_list: tensors+mask'''
        if not isinstance(tensor_list, NestedTensor):
            b, t = tensor_list.shape[0], tensor_list.shape[2]
            mask = torch.zeros((b, t), dtype=torch.bool, device=tensor_list.device)
            tensor_list = NestedTensor(tensor_list, mask)
        tensors = tensor_list.tensors
        batch_size = tensors.shape[0]
        mask = tensor_list.mask
        shape = tensors.shape

        # it takes as input image sequence or feature vector sequence
        if len(shape) == 5:   # (n,c,t,h,w)
            pooler = F.adaptive_max_pool3d if cfg.spatial_pool == 'max' else F.adaptive_avg_pool3d

            ip = tensor_list.tensors
            if cfg.snippet_wise_feature:
                ip = unfold(tensor_list.tensors, cfg.snippet_length, cfg.snippet_stride)
                video_ft = self.backbone(ip).mean(2)       # (n*n_window, c, t, h, w)
                T = video_ft.shape[0] // batch_size
                video_ft_fold = video_ft.reshape(batch_size, T, *(video_ft.shape[1:]))  # (n, n_window, c, h, w)
                video_ft = video_ft_fold.transpose(1, 2)
            else:
                # fully convolutional feature extraction
                video_ft = self.backbone(tensor_list.tensors)  # [n,c,t, h, w]

            video_ft = self.neck(video_ft)
            # if isinstance(video_ft, (list, tuple)) and len(video_ft) == 1:
            #     video_ft = video_ft[0]

            if not isinstance(video_ft, (list, tuple)):
                if video_ft.ndim == 5:
                    video_ft = pooler(video_ft, [None, 1, 1])[..., 0, 0]  # [n, c, t]
                if self.use_upsample:
                    video_ft = F.interpolate(video_ft, scale_factor=cfg.temporal_upscale, mode='linear')
                mask = F.interpolate(mask[None].float(), size=video_ft.shape[2], mode='nearest').to(torch.bool)[0]  # [n, t]
                out = NestedTensor(video_ft, mask)
            else:
                # multilevel feature from backbone
                mask = F.interpolate(mask[None].float(), size=video_ft[0].shape[2], mode='nearest').to(torch.bool)[0]  # [n, t]
                out: List[NestedTensor] = [
                    NestedTensor(x, mask)
                    for x in video_ft
                ]
                # raise NotImplementedError

        elif len(shape) == 3: # (n,c,t)
            video_ft = tensors
            out = NestedTensor(video_ft, mask)

        return out

    def _fix_encoder(self):
        logging.info('freezing the backbone')
        self.backbone.requires_grad_(False)


class EmptyEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.num_channels = feature_dim

    def forward(self, x):
        return x


def build_video_encoder(args):
    if args.input_type == 'feature':
        model = EmptyEncoder(args.feature_dim)
    else:
        model = VideoEncoder(args.encoder, args.fix_encoder, args.neck)
    return model
