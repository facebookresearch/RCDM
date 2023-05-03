"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
import os
import torch as th
import torch.nn as nn
from torch.cuda.amp import autocast
from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.get_ssl_models import get_model
from guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from guided_diffusion_rcdm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from guided_diffusion_rcdm.image_datasets import load_single_image

class RCDM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        # Init SSL model
        self.ssl_model = get_model(args.type_model, args.use_head).cuda().eval()
        for p in self.ssl_model.parameters():
            self.ssl_model.requires_grad = False
        self.ssl_dim = self.ssl_model(th.zeros(1,3,224,224).cuda()).size(1)
        # Init RCDM
        self.rcdm_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=False, feat_cond=True, ssl_dim=self.ssl_dim
        )
        # Load model
        if args.model_path == "":
            trained_model = get_dict_rcdm_model(args.type_model, args.use_head)
        else:
            trained_model = th.load(args.model_path, map_location="cpu")
        self.rcdm_model.load_state_dict(trained_model, strict=True)
        self.rcdm_model.to(dist_util.dev())
        self.rcdm_model.eval()
        # Set sample function
        self.sample_fn = (self.diffusion.p_sample_loop if not args.use_ddim else self.diffusion.ddim_sample_loop)

    @th.no_grad()
    def forward(self, batch, num_samples=1):
        model_kwargs = {}
        # Load the image if path is given
        if type(batch) is str and os.path.exists(batch):
            batch = th.from_numpy(load_single_image(batch)).cuda().unsqueeze(0)
        # Must process one image at a time
        with autocast():
            list_samples = []
            for _ in range(batch.size(0)):
                # Generate num_samples from this representation
                feat = self.ssl_model(batch).detach()
                model_kwargs["feat"] = feat
                sample = self.sample_fn(
                    self.rcdm_model,
                    (num_samples, 3, self.args.image_size, self.args.image_size),
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                )
                list_samples.append(sample)
        return th.cat(list_samples, dim=0)
