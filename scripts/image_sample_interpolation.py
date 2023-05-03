"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""

import argparse
import numpy as np
import torch as th
from PIL import Image
import blobfile as bf
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from guided_diffusion_rcdm.image_datasets import load_data
from guided_diffusion_rcdm import dist_util, logger
from guided_diffusion_rcdm.get_ssl_models import get_model
from guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from guided_diffusion_rcdm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def exclude_bias_and_norm(p):
        return p.ndim == 1

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=6):
	# interpolate ratios between the points
	ratios = np.linspace(0.5, 0.7, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

def main(args):
    args.gpu = 0
    logger.configure(dir=args.out_dir)

    # Use features conditioning
    ssl_model = get_model(args.type_model, args.use_head).cuda().eval()
    for p in ssl_model.parameters():
        ssl_model.requires_grad = False
    ssl_dim = ssl_model(th.zeros(1,3,224,224).cuda()).size(1)
    tr_normalize = transforms.Normalize(
            mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]
        )

    # ============ preparing model ... ============
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=True, ssl_dim=ssl_dim
    )

    # Load model
    if args.model_path == "":
        trained_model = get_dict_rcdm_model(args.type_model, args.use_head)
    else:
        trained_model = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(trained_model, strict=True)
    model.to(dist_util.dev())
    model.eval()

    # Define transforms
    tr_normalize = transforms.Normalize(
            mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]
        )
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        tr_normalize,
    ])
    transform_small = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        tr_normalize,
    ])

    ### Choose first image
    with bf.BlobFile(args.first_image_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    first_image = transform(pil_image).unsqueeze(0)
    first_image_small = transform_small(pil_image).unsqueeze(0)

    # Compute its embedding
    with th.no_grad():
        feat_1 = ssl_model(first_image.cuda()).detach()
    
    ### Choose second image
    with bf.BlobFile(args.second_image_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    second_image = transform(pil_image).unsqueeze(0)
    second_image_small = transform_small(pil_image).unsqueeze(0)

    # Compute its embedding
    with th.no_grad():
        feat_2 = ssl_model(second_image.cuda()).detach()
    
    # Perform an interpolation in the representation space
    inp_features = th.FloatTensor(interpolate_points(feat_1.cpu().numpy(), feat_2.cpu().numpy(), n_steps=args.batch_size)).cuda().view(args.batch_size,-1)

    first_image = ((first_image_small + 1) * 127.5).clamp(0, 255).to(th.uint8)
    first_image = first_image.permute(0, 2, 3, 1)
    first_image = first_image.contiguous()

    second_image = ((second_image_small + 1) * 127.5).clamp(0, 255).to(th.uint8)
    second_image = second_image.permute(0, 2, 3, 1)
    second_image = second_image.contiguous()

    all_images = []

    # We keep the noise fixed
    noise = th.randn((1,3,args.image_size,args.image_size)).cuda()
    noise = noise.repeat(args.batch_size, 1, 1, 1)
    model_kwargs = {}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    model_kwargs["feat"] = inp_features
    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        noise=noise,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )

    all_images.extend([first_image.cpu().numpy()])

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    samples = sample.contiguous()

    all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
    all_images.extend([second_image.cpu().numpy()])
    logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    save_image(th.FloatTensor(arr).permute(0,3,1,2), args.out_dir+'/'+args.name+'.jpeg', normalize=True, scale_each=True, nrow=args.batch_size+2)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_images=1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        submitit=False,
        local_rank=0,
        dist_url="env://",
        G_shared=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="interpolation", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--first_image_path', default=".", type=str, help='Path of the first image to interpolate from.')
    parser.add_argument('--second_image_path', default=".", type=str, help='Path of the second image to interpolate from.')   
    parser.add_argument('--feat_cond', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='Use the projector/head to compute the SSL representation instead of the backbone.')
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Compute distance ssl representation.')
    parser.add_argument('--type_model', type=str, default="dino",
                    help='Select the type of model to use.')

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
