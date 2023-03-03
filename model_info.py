import torch
from ptflops import get_model_complexity_info

from models.spatial_pooler import TemporalWiseAttentionPooling
from models.transformer_ori import DeformableTransformer
from models.video_encoder_archs.slowfast import ResNet3dSlowFast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_gpu_memory_consumption(model, input):
    before = torch.cuda.max_memory_allocated()
    x = model(input)
    x.mean().backward()
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1024 / 1024


# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2012. Xiaolong Liu
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
'''Entry for training and testing'''

import datetime
import json
import logging
import os
import os.path as osp
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import test, train_one_epoch
from models import build_model
from opts import cfg, get_args_parser, update_cfg_from_file, update_cfg_with_args

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.cfg is not None:
        update_cfg_from_file(cfg, args.cfg)

    update_cfg_with_args(cfg, args.opt)


    with torch.autograd.profiler.profile(with_flops=True, with_modules=True) as prof:
        before = torch.cuda.max_memory_allocated()
        model, criterion, postprocessors = build_model(cfg)
        if not args.resume and not args.eval and cfg.input_type == 'image':
            model.backbone.backbone.load_pretrained_weight(cfg.pretrained_model)

        model.cuda()
        # with torch.no_grad():
        #     print(f'The number of parameters: {count_parameters(backbone) / 1000 / 1000} M')

        input = torch.randn(4, 3, 256, 224, 224).cuda()
        mask = torch.ones(4, 256, dtype=torch.bool).cuda()
        x = model((input, mask))

        after = torch.cuda.max_memory_allocated()
    # macs, params = get_model_complexity_info(model, (3, 256, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    stats = prof.key_averages().table()
    index = stats.find("FLOPs")
    unit = stats[index - 1]
    print(unit)
    flops = 0
    for k in prof.key_averages():
        flops += k.flops
    print(f'The number of GPU meory: {(after - before) / 1024 / 1024 / 1024} GB')
    print(f'Total FLOPs: {flops / 1000 / 1000}M')