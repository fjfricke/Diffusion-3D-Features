import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np


def init_sam2(device, model_size="large"):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # model = model.to(device).eval()c\
    return model


torch.no_grad()
def get_sam_features(device, sam_model, img, grid):
   # Use SAM's predictor to get features
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(img)
    # Get features from the specified intermediate layer
    features = predictor._features["image_embed"]
    # features = predictor._features["high_res_feats"][0]

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
