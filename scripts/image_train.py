"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
Train a conditional (representation based) diffusion model on images.
"""

import argparse
import torch as th
import torch.nn as nn
from guided_diffusion_rcdm import dist_util, logger
from guided_diffusion_rcdm.image_datasets import load_data
from guided_diffusion_rcdm.resample import create_named_schedule_sampler
from guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_rcdm.train_util import TrainLoop
from guided_diffusion_rcdm.get_ssl_models import get_model
from torch.cuda.amp import autocast

def main(args):
    # Init distributed setup
    dist_util.init_distributed_mode(args)
    logger.configure(dir=args.out_dir)

    # Load SSL model
    if args.feat_cond:
        ssl_model = get_model(args.type_model, args.use_head).to(args.gpu).eval()
        ssl_dim = ssl_model(th.zeros(1,3,224,224).to(args.gpu)).size(1)
        print("SSL DIM:", ssl_dim)
        for _,p in ssl_model.named_parameters():
            p.requires_grad_(False)
    else:
        ssl_model = None
        ssl_dim = 2048
        print("No SSL models")

    # Create RCDM
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=args.feat_cond, ssl_dim=ssl_dim
    )

    model.to(args.gpu)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Create the dataloader
    logger.log("creating data loader...")
    data = load_ssl_data(
        args,
        ssl_model=ssl_model,
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()

def load_ssl_data(args, ssl_model=None):
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond
    )
    for batch, batch_big, model_kwargs in data:
        # We add the conditioning in conditional mode
        if ssl_model is not None:
            with th.no_grad():
                with autocast(args.use_fp16):
                    # we always use an image of size 224x224 for conditioning
                    model_kwargs["feat"] = ssl_model(batch_big.to(args.gpu)).detach()
            yield batch, model_kwargs
        else:
            yield batch, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        submitit=False,
        local_rank=0,
        dist_url="env://",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--feat_cond', action='store_true', default=False,
                        help='Activate conditional RCDM.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='Disable the shared lower dimensional projection of the representation.')
    parser.add_argument('--type_model', type=str, default="dino",
                    help='Select the type of model to use.')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
