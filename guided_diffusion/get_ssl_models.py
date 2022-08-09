# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchvision import models as torchvision_models

class Wrapper(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, model, head, use_head=False):
        super().__init__()
        self.head = head
        self.model = model
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.use_head = use_head

    def forward(self, x):
        x = self.model(x)
        if x.ndim > 2:
            x = self.pooling(x).view(x.size(0), -1)
        if self.use_head:
            x = self.head(x)
        return x

class SimCLRHead(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        return x

def Projector(emb=8192):
    mlp_spec = f"2048-{8192}-{8192}-{emb}"

    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

class Normalize(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2).detach()
        return x

def get_model(model="dino", use_head=False):
    '''
    Select a model that will be used to compute the embeddings needed by RCDM.
    You can use any kind of model, ConvNets/MLPs, or VITs.
    '''

    if model == "supervised":
        embedding_model = torchvision_models.resnet50(pretrained=True)
        embedding_model.fc = nn.Identity()
        return embedding_model.eval()

    elif model == "dino":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        embedding_model.head = DINOHead(
                2048,
                60000,
                nlayers=2,
                use_bn=True,
        )
        pretrained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth", map_location="cpu") 
        pretrained_model = pretrained_model["teacher"]
        if "state_dict" in pretrained_model:
            pretrained_model = pretrained_model["state_dict"]
        # remove prefixe "module."
        pretrained_model = {k.replace("module.", ""): v for k, v in pretrained_model.items()}
        pretrained_model = {k.replace("backbone.", ""): v for k, v in pretrained_model.items()}
        for k, v in embedding_model.state_dict().items():
            if k not in list(pretrained_model):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif pretrained_model[k].shape != v.shape:
                print(pretrained_model[k].shape, "/", v.shape)
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                pretrained_model[k] = v
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print(msg)
        return Wrapper(embedding_model, head=embedding_model.head, use_head=use_head)

    elif model == "simclr":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch", map_location="cpu")
        # Load trunk
        pretrained_model = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["trunk"]
        pretrained_model = {k.replace("_feature_blocks.", ""): v for k, v in pretrained_model.items()}
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print("Trunk:", msg)
        # Load head
        embedding_model_head = SimCLRHead(2048, hidden_dim=2048, bottleneck_dim=128)
        pretrained_model_head = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["heads"]
        pretrained_model_head = {k.replace("0.clf.0", "mlp.0").replace("1.clf.0", "mlp.1"): v for k, v in pretrained_model_head.items()}
        msg = embedding_model_head.load_state_dict(pretrained_model_head, strict=True)
        embedding_model_head.mlp = nn.Sequential(embedding_model_head.mlp[0], nn.ReLU(), embedding_model_head.mlp[1])
        embedding_model.eval()
        print("Head:", msg)
        return Wrapper(embedding_model, embedding_model_head, use_head)

    elif model == "barlow":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch", map_location="cpu")
        # Load trunk        
        pretrained_model = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["trunk"]
        pretrained_model = {k.replace("_feature_blocks.", ""): v for k, v in pretrained_model.items()}
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print("Trunk:", msg)
        # Load head
        head1 = nn.Sequential(nn.Linear(2048, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU())
        head2 = nn.Sequential(nn.Linear(8192, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU())
        embedding_model.head = nn.Sequential(head1, head2, nn.Sequential(nn.Linear(8192, 8192, bias=False)))
        pretrained_model_head = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["heads"]
        pretrained_model_head = {k.replace("clf.", ""): v for k, v in pretrained_model_head.items()}
        msg = embedding_model.head.load_state_dict(pretrained_model_head, strict=True)
        print("Head:", msg)
        embedding_model.eval()
        return Wrapper(embedding_model, embedding_model.head, use_head)

    elif model == "vicreg":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vicreg/resnet50_fullckpt.pth", map_location="cpu")
        embedding_model.classifier = nn.Identity()
        embedding_model.projector = Projector(emb=8192)
        pretrained = "resnet50_fullckpt.pth"
        pretrained_model_base["model"] = {k.replace("backbone.", ""): v for k, v in pretrained_model_base["model"].items()}
        pretrained_model_base["model"] = {k.replace("module.", ""): v for k, v in pretrained_model_base["model"].items()}
        msg = embedding_model.load_state_dict(pretrained_model_base["model"], strict=True)
        embedding_model.eval()
        print(msg)
        return Wrapper(embedding_model, embedding_model.projector, use_head)

    else:
        print("No model found")
        exit(1)
