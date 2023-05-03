"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""

import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

from tqdm import tqdm
import faiss
from PIL import Image
import blobfile as bf
from guided_diffusion_rcdm.image_datasets import ImageDataset, _list_image_files_recursively
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

def generate_embeddings(model, device, loader, mlp=False):
    '''
    Function to embed the data from the pixel space to a representation space induced by model.
    Saves the embeddings and also return them as output along with the respective labels.
    '''
    embeddings = []

    iterator = tqdm(loader, total=len(loader))
    with th.no_grad():
        for batch in iterator:
            small, x, y = batch
            x = x.to(device)
            emb = model(x)
            emb = emb.view(emb.size(0), -1)
            if mlp:
                emb = model.head.mlp(emb)
            emb = th.nn.functional.normalize(emb, p=2, dim=-1)

            embeddings.append(emb.detach().cpu())

    embeddings = th.cat(embeddings, 0).numpy()

    return embeddings

def nearest_neighbors_faiss(embeddings, target, device):
    k = 20 # We use 20 nearest neighbors
    d = embeddings.shape[-1] 
    index_flat = faiss.IndexFlatL2(d)

    if 'cuda' in str(device):
        print('using gpu')
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        nlist = 100
        index = faiss.IndexIVFFlat(index_flat, d, nlist)
        index.nprobe = 100
    
    index.train(embeddings)
    index.add(embeddings)
    distance, neighbor_idx = index.search(target, k)
    return distance, neighbor_idx

def compute_dist_ssl(ssl_model, sample, target):
    with th.no_grad():
        feat = ssl_model(sample).detach()        
    distance = ((feat - target)**2).sum(1)
    val = np.round(distance.detach().view(-1,1).cpu().numpy(), 1)
    return val

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

    ### Get target image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        tr_normalize,
    ])
    with bf.BlobFile(args.image_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    batch_original = transform(pil_image).unsqueeze(0)
    # Compute its embedding
    with th.no_grad():
        feat_original = ssl_model(batch_original[0:1].cuda()).detach()

    ### Get neighbors folder
    all_files = _list_image_files_recursively(args.folder_nn_path)
    dataset = ImageDataset(
        args.image_size,
        all_files,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )

    # Compute the embeddings of the images that will use to find knn
    embeddings = generate_embeddings(ssl_model, args.gpu, loader, mlp=False)
    # Then compute nearest_neighbors of the target image
    distance, index = nearest_neighbors_faiss(embeddings, feat_original.cpu().numpy(), "cuda:0")
    list_voisins = []
    list_feat = []
    # Creage a list of the embeddings of the nearest neigbors
    for k in range(len(index[0])):
        # We took only the non zero dimensions
        list_feat.append(th.nonzero(th.from_numpy(embeddings[index[0][k]])))
    list_feat = th.sort(th.flatten(th.cat(list_feat, dim=0)))[0]

    # Then we compute a count of how many times a given dimension is non zero accross the neigborhood
    lf, c = th.unique(list_feat, return_counts=True, sorted=True)
    #list_voisins = th.cat(list_voisins, dim=0)
    logger.log("sampling...")
    all_images = []
    noise = th.randn((args.batch_size,3,128,128)).cuda().repeat(6,1,1,1)
    noise_steps = [th.randn_like(noise) for i in range(diffusion.num_timesteps)]

    ### Get the image on which we will apply the transformation on
    with bf.BlobFile(args.image_target_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    batch_target = transform(pil_image).unsqueeze(0)
    # Compute its embedding
    with th.no_grad():
        feat_target = ssl_model(batch_target[0:1].cuda()).detach()

    ### Run RCDM with the new representations
    model_kwargs = {}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    with th.no_grad():
        feat = feat_original.repeat(args.batch_size, 1)

        feat_zero = feat.clone()
        feat_zero[:, lf[(c > args.n_common)]] = 0

        feat_transformation = feat.clone()
        feat_transformation[:, lf[(c > args.n_common)]] = feat_target[:, lf[(c > args.n_common)]]

        model_kwargs["feat"] = th.cat((feat, feat_zero, feat_transformation))

    sample = sample_fn(
        model,
        (model_kwargs["feat"].size(0), 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    samples = sample.contiguous()

    all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
    logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)        
    save_image(th.FloatTensor(arr).permute(0,3,1,2), args.out_dir+'/'+args.name+'.jpeg', normalize=True, scale_each=True, nrow=args.batch_size)

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
    parser.add_argument('--n_common', default=14, type=int, help='Number of common dim to remove')
    parser.add_argument('--name', default="manipulation", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--image_path', default=".", type=str, help='Path of the image you want to extract the attributes from.')
    parser.add_argument('--folder_nn_path', default=".", type=str, help='Path of the folder of images to use to compute nearest neigbords.')
    parser.add_argument('--image_target_path', default=".", type=str, help='Path of the image on which to apply the transformation.')
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
