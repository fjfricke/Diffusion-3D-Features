import torch
# import numpy as np
from torchvision import transforms as tfs

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

patch_size = 14

def init_dino(device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
    )
    
    model = model.to(device).eval()
    return model

@torch.no_grad
def get_dino_features(device, dino_model, img, grid):
    transform = tfs.Compose(
        [
            tfs.Resize((518, 518)),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).reshape(1, 768, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    return features

def init_sam(device):

    sam2_checkpoint = "sam2_download/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # model = model.to(device).eval()c\
    return model

@torch.no_grad
def get_sam_features(device, sam_model, img, grid):
    # Use SAM's predictor to get features
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(img)
    # Get features from the specified intermediate layer
    features = predictor._features["image_embed"]
    
    # Process features similar to DINO
    h, w = features.shape[2], features.shape[3]  # Get spatial dimensions directly from features
    dim = features.shape[1]  # Feature dimension is in channel position
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = features.half()
    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).reshape(1, dim, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    return features
