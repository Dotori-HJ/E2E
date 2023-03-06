import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import util.misc as utils
from datasets import build_dataset
from models import build_model
from opts import cfg, get_args_parser, update_cfg_from_file, update_cfg_with_args


def main(args):

    if args.cfg is not None:
        update_cfg_from_file(cfg, args.cfg)

    update_cfg_with_args(cfg, args.opt)

    # The actionness regression module requires CUDA support
    # If your machine does not have CUDA enabled, this module will be disabled.
    if cfg.disable_cuda:
        cfg.act_reg = False

    utils.init_distributed_mode(args)


    mode = 'test'
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Model
    model, criterion, postprocessors = build_model(cfg)
    if not args.resume and not args.eval and cfg.input_type == 'image':
        model.backbone.backbone.load_pretrained_weight(cfg.pretrained_model)

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    elif args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    # Dataset
    dataset_val, _ = build_dataset(subset=cfg.test_set, args=cfg, mode='val')
    if not args.eval:
        dataset_train, gpu_transforms = build_dataset(subset='train', args=cfg, mode='train')
    if args.distributed:
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    data_loader_val = DataLoader(dataset_val, int(cfg.batch_size * 4), sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    for (samples, targets) in tqdm(data_loader_val, total=len(data_loader_val)):
        samples = samples.to(device)
        # outputs, _ = model((samples.tensors, samples.mask))
        outputs = model((samples.tensors, samples.mask))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)