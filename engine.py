# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import logging
import math
import os.path as osp
import pickle
import sys
from typing import Iterable

import torch
import tqdm

import util.misc as utils
from datasets.tad_eval import TADEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cfg, max_norm: float = 0, gpu_transforms=None, use_dn=False, model_ema: torch.nn.Module=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    cnt = 0

    from thop import profile
    import numpy as np
    import time
    import torch.autograd.profiler as profiler
    total_flops = 0
    flops_list = []
    flops_per_video = {}
    current_video_id = None
    current_samples = []
    inf_times = []
    for (samples, targets) in tqdm.tqdm(data_loader):
        # samples = samples.to(device)
        # model((samples.tensors, samples.mask))
        # st = time.time()

        # inf_time = time.time() - st
        # print(inf_time * 1000)
        # exit()
        video_id = '_'.join(targets[0]['video_id'].split('_')[:3])
        if current_video_id is None or current_video_id == video_id:
            current_video_id = video_id
            current_samples.append(samples)
        else:
            if len(current_samples) > 0:
                tensors = torch.cat([x.tensors for x in current_samples], dim=0).to(device)
                masks = torch.cat([x.mask for x in current_samples], dim=0).to(device)
                
                # with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
                #     model((tensors, masks))
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                # exit()
                
                model((tensors, masks))
                st = time.time()
                _, inf_time = model((tensors, masks))
                # inf_time = time.time() - st
                inf_times.append(inf_time)
                
                # flops, params = profile(model, inputs=((tensors, masks),))
                # if video_id not in flops_per_video:
                #     flops_per_video[current_video_id] = flops

            current_video_id = video_id
            current_samples = []

        # flops_per_video[video_id] += flops
            # flops_list.append(flops)


    import numpy as np
    inf_times = np.array(inf_times) * 1000
    print(f"Mean times: {inf_times.mean():.8f}, Std times: {inf_times.std():.8f}")
    exit()
    flops = total_flops / 1e9  # GigaFLOPs로 변환
    params = params / 1e6  # Millions로 변환
    flops_list = list(flops_per_video.values())
    print(f"FLOPs: {flops:.2f} GFLOPs, Params: {params:.2f} M")
    print(f"Min FLOPs: {min(flops_list)/ 1e9:.2f}, Min FLOPs: {max(flops_list)/ 1e9:.2f}")
    print(f"Mean FLOPs: {np.array(flops_list).mean()/ 1e9:.2f}, Std FLOPs: {np.array(flops_list).std()/ 1e9:.2f}")
    print(f"Median FLOPs: {np.median(np.array(flops_list))/ 1e9:.2f}")
    exit()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        if gpu_transforms is not None:
            samples.tensors = gpu_transforms(samples.tensors)
        targets = [{k: v.to(device) if k in ['segments', 'labels']
                    else v for k, v in t.items()} for t in targets]


        if samples.tensors.size(2) != 384:
            with open('t.txt', 'wt') as f:
                f.write(f"{samples.tensors.size()}, {targets}")
        if use_dn:
            scalar = 5
            label_noise_scale = 0.2
            segment_noise_scale = 0.4
            contrastive = True
            dn_args = (targets, scalar, label_noise_scale, segment_noise_scale, 0, contrastive)
            outputs, mask_dict = model((samples.tensors, samples.mask), dn_args=dn_args)
            loss_dict = criterion(outputs, targets, mask_dict)
        else:
            outputs = model((samples.tensors, samples.mask))
            loss_dict = criterion(outputs, targets)

        # outputs = model((samples.tensors, samples.mask))
        # loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss of each type
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # weighted_loss of each type
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logging.info("Loss is {}, stopping training".format(loss_value))
            logging.info(str(loss_dict_reduced))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if (cnt + 1) % cfg.iter_size == 0:
            # scale gradients when iter size is functioning
            if cfg.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= cfg.iter_size

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        cnt += 1

    optimizer.zero_grad()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def to_device(t, device):
    if isinstance(t, (list, tuple)):
        return t
    else:
        return t.to(device)


@torch.no_grad()
def test(model, criterion, postprocessor, data_loader, base_ds, device, output_dir, cfg, subset='val', epoch=None, test_mode=False):
    '''
    Run inference and evaluation. Do not compute loss
    test_mode: indicates that we are evaluating specific epoch during testing
    '''
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))

    iou_range = [0.3, 0.4, 0.5, 0.6, 0.7] if cfg.dataset_name == 'thumos14' else [
        num/100 for num in range(50, 100, 5)]
    # logging.info('iou range {}'.format(iou_range))

    # action_evaluator = None
    # nms_mode = ['nms', 'raw'] if cfg.dataset_name == 'activitynet' else ['raw']
    nms_mode = ['nms']
    action_evaluator = TADEvaluator(cfg.dataset_name, subset, base_ds, nms_mode=nms_mode, iou_range=iou_range, epoch=epoch, topk=cfg.postproc_ins_topk)

    # raw_res = []
    cnt = 0
    for (samples, targets) in tqdm.tqdm(data_loader, total=len(data_loader)):
        samples = samples.to(device)
        # outputs, _ = model((samples.tensors, samples.mask))
        outputs = model((samples.tensors, samples.mask))

        # raw_res.append((outputs, targets))
        video_duration = torch.FloatTensor(
            [t["video_duration"] for t in targets]).to(device)
        results = postprocessor(outputs, video_duration, fuse_score=cfg.act_reg)

        res = {target['video_id']: output for target,
               output in zip(targets, results)}
        if action_evaluator is not None:
            action_evaluator.update(res, assign_cls_labels=cfg.binary)
        # if cnt >= 9:
        #     break
        cnt += 1

    # accumulate predictions from all videos
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate(cfg.test_slice_overlap)
        # dump detections
        if test_mode:
            save_path = osp.join(output_dir, 'detection_{}.json')
            action_evaluator.dump_detection(save_path)
        action_evaluator.summarize()

    stats = {}

    if action_evaluator is not None:
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v)
                              for k, v in stats.items() if k.startswith('mAP')])
        logging.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    # with open('raw_outputs.pkl', 'wb') as f:
    #     pickle.dump(raw_res, f)

    return stats
