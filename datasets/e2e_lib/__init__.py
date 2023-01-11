import os

import cv2
import kornia
import kornia.augmentation as KA
import kornia.enhance as KE
import numpy as np
import torch.nn as nn
from PIL import Image


def load_video_frames(frame_dir, start, seq_len, stride=1, fn_tmpl='img_%07d.jpg'):
    '''
    Load a sequence of video frames into memory. 

    Params:
        frame_dir: the directory to the decoded video frames
        start: load image starting with this index. 1-indexed
        seq_len: length of the loaded sub-sequence.
        stride: load one frame every `stride` frame from the sequence.
    Returns:
        Nd-array with shape (T, H, W, C) in float32 precision. T = num // stride
    '''
    frames = []
    if seq_len > 0:
        # load a fixed-length frame sequence
        # for i in range(start + stride // 2, start + seq_len, stride):
        #     img = cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
        #     if  img is None:
        #         # print('failed to load {}'.format(os.path.join(frame_dir, fn_tmpl % i)))
        #         raise IOError(os.path.join(frame_dir, fn_tmpl % i))
        #     # img = img[:, :, [2, 1, 0]]  # BGR => RGB, moved to video_transforms.Normalize
        #     # img = (img/255.)*2 - 1
        #     frames.append(img)
        # frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
        #     for i in range(start + stride // 2, start + seq_len, stride)]
        frames = [Image.open(os.path.join(frame_dir, fn_tmpl % i))
            for i in range(start + stride // 2, start + seq_len, stride)]
    else:
        # load all frames
        num_imgs = len(os.listdir(frame_dir))
        # frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % (i+1))) for i in range(num_imgs)]
        frames = [Image.open(os.path.join(frame_dir, fn_tmpl % (i+1))) for i in range(num_imgs)]
    if isinstance(frames[0], np.ndarray):
        return np.asarray(frames, dtype=np.float32)  # NHWC
    else:
        return frames


def make_img_transform(is_training, resize=110, crop=96, mean=127.5, std=127.5, keep_asr=True):
    from torchvision.transforms import Compose

    from .videotransforms import (
        GroupCenterCrop,
        GroupNormalize,
        GroupPhotoMetricDistortion,
        GroupRandomCrop,
        GroupRandomHorizontalFlip,
        GroupResize,
        GroupResizeShorterSide,
        GroupRotate,
    )

    if isinstance(resize, (list, tuple)):
        resize_trans = GroupResize(resize)
    else:
        if keep_asr:
            assert isinstance(resize, int), 'if keep asr, resize must be a single integer'
            resize_trans = GroupResizeShorterSide(resize)
        else:
            resize_trans = GroupResize((resize, resize))

    # if is_training:
    #     transforms = [GroupRotate(limit=(-45, 45), border_mode='reflect101', p=0.5)]
    # else:
    #     transforms = []
    transforms = [
        resize_trans,
        GroupRandomCrop(crop) if is_training else GroupCenterCrop(crop),
    ]
    if is_training:
        transforms += [
            GroupRotate(limit=(-45, 45), border_mode='reflect101', p=0.5),
            GroupPhotoMetricDistortion(brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
                p=0.5),
            GroupRandomHorizontalFlip(0.5),
        ]

    if is_training:
        transforms.append(GroupNormalize(127.5, 127.5, to_rgb=True))
    else:
        transforms.append(GroupNormalize(mean, std, to_rgb=True))

    gpu_transforms = None
    # gpu_transforms = GPUAugment([
    #     KA.RandomAffine(30, translate=0.1, shear=0.3, p=1, padding_mode='reflection', same_on_batch=True),
    #     KA.ColorJiggle(0.125, 0.5, 0.5, 0.1, p=1, same_on_batch=True),
    #     KA.RandomHorizontalFlip(p=0.5, same_on_batch=True),
    #     KE.Normalize(mean=torch.tensor(mean) / 255, std=torch.tensor(std) / 255),
    # ])
    # gpu_transform = kornia.augmentation
    return Compose(transforms), gpu_transforms
import torch
from einops import rearrange


class GPUAugment:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensors):
        tensors = rearrange(tensors, "b c t h w -> b t c h w")
        # tensors = [rearrange(x, "c t h w -> t c h w") for x in tensors]
        for op in self.transforms:
            tensors = torch.stack([op(x) for x in tensors])

        tensors = rearrange(tensors, "b t c h w -> b c t h w")
        return tensors