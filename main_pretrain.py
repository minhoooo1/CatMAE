# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
assert timm.__version__ == "0.4.12"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_catmae
from engine_pretrain import train_one_epoch
from kinetics_dataset import CustomKinetics400Dataset, TripRandomResizedCrop

import torch.distributed as dist
import torch.multiprocessing as mp

def get_args_parser():
    parser = argparse.ArgumentParser('CatMAE pre-training', add_help=False)
    parser.add_argument('--pretrain_name', default="catmae-1600ep", type=str, help="branch-commit")
    parser.add_argument('--config_file', type=str, default="", help="config file path")
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--per_save_epochs', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='catmae_vit_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--patch_size', default=16, type=int,
                        help='ViT patch size')
    
    parser.add_argument('--decoder_dim_dep_head', default=[192, 4, 3], type=list,
                        help='Decoder dim dpeths heads config')
    
    parser.add_argument('--input_size', default=(224, 224), type=tuple,
                        help='images input size')

    parser.add_argument('--mask_ratios', default=[0.95, 0.95], type=list,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=16, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--rec_weights', default=[0.8, 1.0], type=float, help='f3 rec weight')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/cv/data/Kinetics-400/videos_train/', type=str,
                        help='dataset path')
    parser.add_argument('--frame_interval', default=[[4,16],[4,48]], type=list)
    parser.add_argument('--repeated_sampling', default=2, type=int)
    
    parser.add_argument('--output_dir', default="./pretrain/",
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default="./pretrain/",
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--gpus', default="0,1", type=str)
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    
    # args.pretrain_name = config.get("pretrain_name", args.pretrain_name)
    # args.batch_size = config.get("batch_size", args.batch_size)
    # args.gpus = config.get("gpus", args.gpus)
    # args.epochs = config.get("epochs", args.epochs)
    # args.warmup_epochs = config.get("warmup_epochs", args.warmup_epochs)
    # args.per_save_epochs = config.get("per_save_epochs", args.per_save_epochs)
    # args.mask_ratios = config.get("mask_ratios", args.mask_ratios)
    # args.rec_weights = config.get("rec_weights", args.rec_weights)
    # args.frame_interval = config.get("frame_interval", args.frame_interval)
    # args.decoder_dim_dep_head = config.get("decoder_dim_dep_head", args.decoder_dim_dep_head)
    # args.output_dir = config.get("output_dir", args.output_dir)
    # args.log_dir = config.get("log_dir", args.log_dir)
    
    for key in config.keys():
        if hasattr(args, key):
            setattr(args, key, config[key])
        else:
            raise ValueError(f"Key '{key}' found in config file is not a valid argument")
    
    
    
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    args.output_dir = args.output_dir + formatted_now + "_" + args.pretrain_name
    args.log_dir = args.log_dir + formatted_now + "_" + args.pretrain_name
    
    args_dict = vars(args)

    os.makedirs(args_dict["log_dir"], exist_ok=True)
    with open(f'{args_dict["log_dir"]}/arg_params.json', 'w') as f:
        json.dump(args_dict, f, indent=2)  

    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '18879'  
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  
    world_size = torch.cuda.device_count() 
    os.environ['WORLD_SIZE'] = str(world_size)

    return args

def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)  
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def main(rank, args):
    args.distributed = True
    init_ddp(rank)  
    args.device = torch.device(f"cuda:{rank}")
    print(f"Start running basic DDP example on rank {rank}.")
    
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # RandomResizeCrop and Hflip augmentation
    transform_triple = TripRandomResizedCrop(size=args.input_size)
    transform_totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = CustomKinetics400Dataset(args.data_path ,transform_triple, transform_totensor, args.frame_interval, args.repeated_sampling)
    

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = rank
        print(f"\n here :{global_rank} num_tasks :{num_tasks}\n")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    
    # define the model
    model = models_catmae.__dict__[args.model](args.patch_size, *args.decoder_dim_dep_head, norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.per_save_epochs == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    try:
        args = get_args_parser()
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # main(args)
        mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    except Exception as e:
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(e)
        with open("error_log.txt", "a") as f:
            error_message = f"{time_str}\nError: {str(e)}\n\n"
            f.write(error_message)
