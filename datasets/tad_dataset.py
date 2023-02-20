# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2022. Xiaolong Liu.
# ------------------------------------------------------------------------

'''Universal TAD Dataset loader.'''

import json
import logging
import math
import os.path as osp

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

# from util.config import cfg
from einops import rearrange, repeat
from PIL import Image

import datasets.video_transforms as video_transforms
import datasets.volume_transforms as volume_transforms
from util.segment_ops import segment_t1t2_to_cw

from .data_utils import get_dataset_dict, get_dataset_info, load_feature
from .e2e_lib import load_video_frames, make_img_transform
from .random_erasing import RandomErasing
from .utils import spatial_sampling, tensor_normalize
from .video_transforms import create_random_augment


class TADDataset(torch.utils.data.Dataset):
    def __init__(self, subset, mode, feature_info, ann_file, ft_info_file, transforms,
                 mem_cache=False, online_slice=False, slice_len=None, slice_overlap=0,
                 binary=False, padding=True, input_type='feature', img_stride=1,
                 resize=256, crop_size=224, rand_augment_param=None, fix_transform=False, rand_erase=False):
        '''TADDataset
        Parameters:
            subset: train/val/test
            mode: train, or test
            feature_info: basic info of video features, e.g. path, file format, filename template
            ann_file: path to the ground truth file
            ft_info_file: path to the file that describe other information of each video
            transforms: which transform to use
            mem_cache: cache features of the whole dataset into memory.
            binary: transform all gt to binary classes. This is required for training a class-agnostic detector
            padding: whether to pad the input feature to `slice_len`

        '''

        super().__init__()
        self.feature_info = feature_info
        self.ann_file = ann_file
        self.ft_info_file = ft_info_file
        self.subset = subset
        self.online_slice = online_slice
        self.slice_len = slice_len
        self.slice_overlap = slice_overlap
        self.padding = padding
        self.mode = mode
        self.transforms = transforms
        # print('Use data transform {}'.format(self.transforms))
        self.binary = binary
        self.is_image_input = input_type == 'image'
        self.mem_cache = mem_cache
        self.img_stride = img_stride

        # self.short_side_size = 110
        # self.crop_size = 96
        self.short_side_size = resize
        self.crop_size = crop_size
        self.rand_augment_param = rand_augment_param
        self.rand_erase = rand_erase
        self._prepare()
        # if mode == 'train':
        #     if fix_transform:
        #         self.transforms = self._test_transform
        #         # self.transforms = video_transforms.Compose([
        #         #     video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
        #         #     # video_transforms.CenterCrop(size=(self.short_side_size, self.short_side_size)),
        #         #     video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
        #         #     volume_transforms.ClipToTensor(),
        #         #     video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #         #                             std=[0.229, 0.224, 0.225])
        #         # ])
        #     else:
        #         self.transforms = self._train_transform
        #         # self.train_transforms = video_transforms.Compose([
        #         #     video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
        #         #     video_transforms.RandomCrop(self.crop_size),
        #         #     video_transforms.RandomHorizontalFlip(),
        #         #     volume_transforms.ClipToTensor(),
        #         #     video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #         #                                std=[0.229, 0.224, 0.225])
        #         # ])
        # else:
        #     self.transforms = self._test_transform
        #     # self.transforms = video_transforms.Compose([
        #     #     video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
        #     #     # video_transforms.CenterCrop(size=(self.short_side_size, self.short_side_size)),
        #     #     video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
        #     #     volume_transforms.ClipToTensor(),
        #     #     video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                                std=[0.229, 0.224, 0.225])
        #     # ])

    def _train_transform(self, imgs):
        transform = create_random_augment(imgs[0].size, self.rand_augment_param, "bilinear")

        imgs = transform(imgs)
        imgs = [transforms.ToTensor()(img) for img in imgs]
        imgs = torch.stack(imgs) # T C H W
        imgs = imgs.permute(0, 2, 3, 1) # T H W C

        # T H W C
        imgs = tensor_normalize(
            imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        imgs = imgs.permute(3, 0, 1, 2)
        # Perform data augmentation.
        # scl, asp = (
        #     [0.8, 1.0],
        #     [0.75, 1.3333],
        # )
        imgs = spatial_sampling(
            imgs,
            spatial_idx=-1,
            min_scale=self.short_side_size,
            max_scale=self.short_side_size,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            # aspect_ratio=asp,
            aspect_ratio=None,
            # scale=scl,
            scale=None,
            motion_shift=False
        )

        return imgs

    def _test_transform(self, imgs):
        imgs = [transforms.ToTensor()(img) for img in imgs]
        imgs = torch.stack(imgs) # T C H W
        imgs = imgs.permute(0, 2, 3, 1) # T H W C

        # T H W C
        imgs = tensor_normalize(
            imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        imgs = imgs.permute(3, 0, 1, 2)
        imgs = spatial_sampling(
            imgs,
            spatial_idx=1,    # center crop
            min_scale=self.short_side_size,
            max_scale=self.short_side_size,
            crop_size=self.crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False,
            aspect_ratio=None,
            scale=None,
            motion_shift=False
        )

        return imgs


    def _get_classes(self, anno_dict):
        '''get class list from the annotation dict'''
        if 'classes' in anno_dict:
            classes = anno_dict['classes']
        else:
            database = anno_dict['database']
            all_gts = []
            for vid in database:
                all_gts += database[vid]['annotations']
            classes = list(sorted({x['label'] for x in all_gts}))
        return classes

    def _prepare(self):
        '''parse annotation file'''
        anno_dict = json.load(open(self.ann_file))
        self.classes = self._get_classes(anno_dict)

        self.video_dict, self.video_list = get_dataset_dict(self.ft_info_file, self.ann_file, None, None, self.subset, mode=self.mode, online_slice=self.online_slice, slice_len=self.slice_len, slice_overlap=self.slice_overlap, ignore_empty=self.mode == 'train', return_id_list=True)

        # video_list = self.video_dict.keys()
        # self.video_list = list(sorted(video_list))

        logging.info("{} subset video numbers: {}".format(self.subset,len(self.video_list)))
        self.anno_dict = anno_dict

        self.remove_duplicated_and_short()

        self.cached_data = {}

        # if the features of all videos is saved in one hdf5 file (all in one), e.g. TSP features
        self.all_video_data = {}
        feature_info = self.feature_info
        fn_templ = feature_info['fn_templ']
        src_video_list = {self.video_dict[k]['src_vid_name'] for k in self.video_list}
        #
        if feature_info.get('all_in_one', False):
            data = h5py.File(feature_info['local_path'][self.subset])
            for k in src_video_list:
                self.all_video_data[k] = np.array(data[fn_templ % k]).T
            if not self.online_slice:
                self.cached_data = self.all_video_data

    def remove_duplicated_and_short(self, eps=0.02):
        num_removed = 0
        for vid in self.anno_dict['database'].keys():
            annotations = self.anno_dict['database'][vid]['annotations']
            valid_annos = []

            for anno in annotations:
                s, e = anno["segment"]
                l = anno["label"]

                if (e - s) >= eps:
                    valid = True
                else:
                    valid = False
                for v_anno in valid_annos:
                    if ((abs(s - v_anno['segment'][0]) <= eps)
                        and (abs(e - v_anno['segment'][1]) <= eps)
                        and (l == v_anno['label'])
                    ):
                        valid = False
                        break

                if valid:
                    valid_annos.append(anno)
                else:
                    num_removed += 1

            self.anno_dict['database'][vid]['annotations'] = valid_annos
        if num_removed > 0:
            print(f"Removed {num_removed} duplicated and short annotations")

    def __len__(self):
        return len(self.video_list)

    def _get_video_data(self, index):
        if self.is_image_input:
            return self._get_img_data(index)
        else:
            return self._get_feature_data(index)

    def _get_feature_data(self,index):
        video_name = self.video_list[index]
        # directly fetch from memory
        if video_name in self.cached_data:
            video_data = self.cached_data[video_name]
            return torch.Tensor(video_data).float().contiguous()

        src_vid_name = self.video_dict[video_name]['src_vid_name']
        # retrieve feature info
        feature_info = self.feature_info
        # "ft" is short for "feature"
        local_ft_dir = feature_info['local_path']
        ft_format = feature_info['format']
        local_ft_path = osp.join(local_ft_dir, feature_info['fn_templ'] % src_vid_name) if local_ft_dir else None
        # the shape of feature sequence, can be TxC (in most cases) or CxT
        shape = feature_info.get('shape', 'TC')

        if src_vid_name in self.all_video_data:
            feature_data = self.all_video_data[src_vid_name].T
        else:
            feature_data = load_feature(local_ft_path, ft_format, shape)

        feature_data = feature_data.T   # T x C to C x T.

        if self.online_slice:
            slice_start, slice_end = [int(x) for x in video_name.split('_')[-2:]]
            assert slice_end  > slice_start
            assert slice_start < feature_data.shape[1]
            feature_data = feature_data[:, slice_start:slice_end]

            if self.padding and feature_data.shape[1] < self.slice_len:
                diff = self.slice_len - feature_data.shape[1]
                feature_data = np.pad(
                    feature_data, ((0, 0), (0, diff)), mode='constant')

                # IMPORATANT: if padded is done, the length info must be modified
                self.video_dict[video_name]['feature_length'] = self.slice_len
                self.video_dict[video_name]['feature_second'] = self.slice_len / self.video_dict[video_name]['feature_fps']

        if self.mem_cache and video_name not in self.cached_data:
            self.cached_data[video_name] = feature_data

        feature_data = torch.Tensor(feature_data).float().contiguous()
        return feature_data

    def _get_img_data(self, index):
        video_name = self.video_list[index]
        src_vid_name = self.video_dict[video_name]['src_vid_name']

        feature_info = self.feature_info

        frame_dir = osp.join(feature_info['local_path'], feature_info['fn_templ'] % src_vid_name)

        if self.online_slice:
            # for THUMOS14
            slice_start, slice_end = [int(x) for x in video_name.split('_')[-2:]]
            start_idx = slice_start

            # clip_length = end_frame_index - start_frame_index + 1. It counts skipped frames when img_stride > 1
            dst_clip_length = self.slice_len
            # clip_length: the argument passed to the img loader
            clip_length = slice_end - slice_start

            imgs = load_video_frames(frame_dir, start_idx + 1, clip_length, self.img_stride)
            assert len(imgs) != 0

            # the actual number of frames
            dst_sample_frames = dst_clip_length // self.img_stride

            if len(imgs) < dst_sample_frames:
                # try:
                # if isinstance(imgs, np.ndarray):
                #     imgs = np.pad(imgs, ((0, dst_sample_frames - len(imgs)), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=128)
                # else:
                #     tmp = Image.new("RGB", imgs[0].size, (128, 128, 128))
                #     imgs += [tmp for i in range(dst_sample_frames - len(imgs))]
                # except:
                #     pdb.set_trace()
                self.video_dict[video_name]['feature_length'] = self.slice_len
                self.video_dict[video_name]['feature_second'] = self.slice_len / self.video_dict[video_name]['feature_fps']
        else:
            start_idx = 0
            video_length = self.video_dict[video_name]['feature_length']
            dst_clip_length = feature_info.get('num_frames', None)
            clip_length = min(video_length, dst_clip_length) if dst_clip_length is not None else video_length

            imgs = load_video_frames(frame_dir, start_idx + 1, clip_length, self.img_stride)

            # On ActivityNet/HACS, we use ffmpeg to decode a video into fixed number of frames.
            # However, the actual number of decoded frames may differ from the desired number.

            if dst_clip_length:
                dst_sample_frames = dst_clip_length // self.img_stride

                # if len(imgs) < dst_sample_frames:
                #     if isinstance(imgs, np.ndarray):
                #         imgs = np.pad(imgs, ((0, dst_sample_frames - len(imgs)), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=128)
                #     else:
                #         tmp = Image.new("RGB", imgs[0].size, (128, 128, 128))
                #         imgs += [tmp for i in range(dst_sample_frames - len(imgs))]

                imgs = imgs[:dst_sample_frames]
        # try:
        #     imgs = self.transforms(imgs)
        # except Exception as e:
        #     # traceback.print_exc()
        #     raise IOError("failed to transform {} from {}".format(video_name, frame_dir))
        imgs = self.transforms(imgs)
        # if self.online_slice:
        #     c, t, h, w = imgs.size()
        #     if t < dst_sample_frames:
        #         imgs = torch.cat((
        #             imgs, torch.zeros(c, dst_sample_frames - t, h, w, dtype=imgs.dtype)
        #         ), dim=1)

        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(np.ascontiguousarray(imgs.transpose([3, 0, 1, 2]))).float()   # thwc -> cthw
        return imgs

    def _get_train_label(self, video_name):
        '''get normalized target'''
        video_info = self.video_dict[video_name]
        video_labels = video_info['annotations']
        feature_second = video_info['feature_second']

        target = {
            'segments': [], 'labels': [],
            'orig_labels': [], 'video_id': video_name,
            'video_duration': feature_second,   # only used in inference
            'feature_fps': video_info['feature_fps'],
            }
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]

            segment = tmp_info['segment']
            # special rule for thumos14, treat ambiguous instances as negatives
            if tmp_info['label'] not in self.classes:
                continue
            # the label id of first forground class is 0
            label_id = self.classes.index(tmp_info['label'])
            target['orig_labels'].append(label_id)

            if self.binary:
                label_id = 0
            target['segments'].append(segment)
            target['labels'].append(label_id)

        # normalized the coordinate
        # if ((np.array(target['segments'])[:, 1] - feature_second) > 0).sum() > 0:
        #     print(video_name, np.array(target['segments'])[:, 1], feature_second)
        target['segments'] = np.array(target['segments']) / feature_second
        target['segments'] = np.clip(target['segments'], 0, 1)
        # if (target['segments'] < 0).sum() + (target['segments'] > 1).sum() > 0:
        # if (target['segments'] < 0).sum():
            # print(f"!!!!!, {video_name} {target['segments']}")

        if len(target['segments']) > 0:
            target['segments'] = segment_t1t2_to_cw(target['segments'])

            # convert to torch format
            for k, dtype in zip(['segments', 'labels'], ['float32', 'int64']):
                if not isinstance(target[k], torch.Tensor):
                    target[k] = torch.from_numpy(np.array(target[k], dtype=dtype))

        return target

    def __getitem__(self, index):
        # index = index % len(self.video_list)
        video_data = self._get_video_data(index)
        video_name = self.video_list[index]

        target =  self._get_train_label(video_name)

        return video_data, target
        # return None, target


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


def build(dataset, subset, args, mode):
    '''build TADDataset'''
    subset_mapping, feature_info, ann_file, ft_info_file = get_dataset_info(dataset, args.feature)
    transforms = None
    if args.input_type == 'image':
        if args.encoder == 'i3d':
            mean, std = (127.5, 127.5)
        elif args.encoder == 'slowfast' or 'video_mae' or 'video_swin' in args.encoder or args.backbone.startswith('ts'):
            mean, std = ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
        is_training = mode == 'train' and not args.fix_transform
        transforms, gpu_transforms = make_img_transform(
            is_training=is_training, mean=mean, std=std, resize=args.img_resize, crop=args.img_crop_size,
            num_frames=args.slice_len, keep_asr=args.resize_keep_asr
        )
        # transforms = None
        # gpu_transforms = GPUAugment([
        #     KA.RandomAffine(30, translate=0.1, shear=0.3, p=0.5, same_on_batch=True),
        #     KA.ColorJiggle(0.125, 0.5, 0.5, 0.1, p=0.5, same_on_batch=True),
        #     KA.RandomHorizontalFlip(p=0.5, same_on_batch=True),
        #     KE.Normalize(mean=torch.tensor(mean) / 255, std=torch.tensor(std) / 255),
        # ])
        # gpu_transforms = None
    else:
        transforms = None
        gpu_transforms = None

    return TADDataset(
        subset_mapping[subset], mode, feature_info, ann_file, ft_info_file, transforms,
        online_slice=args.online_slice, slice_len=args.slice_len, slice_overlap=args.slice_overlap if mode=='train' else args.test_slice_overlap,
        binary=args.binary,
        input_type=args.input_type,
        resize=args.img_resize,
        crop_size=args.img_crop_size,
        rand_augment_param=args.rand_augment_param,
        fix_transform=args.fix_transform,
        rand_erase=args.rand_erase
        ), gpu_transforms
