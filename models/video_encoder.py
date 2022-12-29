"""
Video encoder modules.
"""
import logging
from collections import OrderedDict
from typing import Dict, List

import ipdb as pdb
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn
from torch.nn.modules.normalization import GroupNorm
from torchvision.models._utils import IntermediateLayerGetter

from models.video_encoder_archs.slowfast import ResNet3dSlowFast
from models.video_encoder_archs.tsm import TSM
from models.video_encoder_archs.video_mae import VisionTransformer
from opts import cfg
from util.misc import NestedTensor, is_main_process


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
    def forward(self, x):
        return x[-1]


class TunerBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels

        # Conv
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x))) + x

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

class PyramidTuner(nn.Module):
    def __init__(self, feature_dims:tuple, middle_dim, output_dim):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Conv1d(dim, middle_dim, kernel_size=3)
            for dim in feature_dims
        ])
        self.middle_layers = nn.ModuleList([
            TunerBlock(middle_dim, 2048, middle_dim, kernel_size=3)
            # nn.Conv1d(middle_dim, middle_dim, kernel_size=3)
            for _ in range(len(feature_dims) - 1)
        ])
        self.output_layer = nn.Conv1d(middle_dim, output_dim, kernel_size=3)
        # self.scaler = nn.Parameter(torch.ones(1))

    def forward(self, features):
        proj_features = [layer(x) for x, layer in zip(features, self.proj_layers)]

        for i, layer in enumerate(self.middle_layers):
            if i == 0:
                out = layer(proj_features[i]) + proj_features[i+1]
            else:
                out = layer(out) + proj_features[i+1]
        out = self.output_layer(out) + features[-1]

        return out

class VideoEncoder(nn.Module):
    def __init__(self, arch='slowfast', fix_encoder=False, neck='pyramid'):
        super().__init__()
        self.arch = arch
        self.use_upsample = cfg.temporal_upsample

        if arch == 'slowfast':
            self.backbone = ResNet3dSlowFast(None, depth=cfg.slowfast_depth,freeze_bn=cfg.freeze_bn, freeze_bn_affine=cfg.freeze_affine, slow_upsample=cfg.slow_upsample)
            self.num_channels = 2304

        elif arch in ['tsm', 'tsn']:
            self.backbone = TSM(arch=cfg.tsm_base_model, is_shift=arch=='tsm')
            self.num_channels = self.backbone.out_channels
        elif arch == 'video_mae':
            self.backbone = VisionTransformer()
            self.num_channels = self.backbone.num_channels

        else:
            raise ValueError('Not supported arch: {}'.format(arch))

        self.fix_encoder = fix_encoder

        if fix_encoder:
            self._fix_encoder()

        if neck == 'identity':
            self.neck = IdentityNeck()
        elif neck == 'pyramid':
            self.neck = PyramidTuner((288, 576, 1152, 2304), 512, self.num_channels)
        elif neck == "tuner":
            self.neck = Tuner(288, 2304, 3)
        else:
            assert True, f"neck={neck}"


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
                raise NotImplementedError

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
