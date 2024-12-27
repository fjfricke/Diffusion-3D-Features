import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np

def init_sam2(device, model_size='large'):
    checkpoint = f"./checkpoints/sam2.1_hiera_{model_size}.pt"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_size[0]}.yaml"
    model = build_sam2(model_cfg, checkpoint)
    predictor = SAM2ImagePredictor(model)
    return predictor

@torch.no_grad()
def get_sam2_features(device, sam_model, img, grid):
    if torch.is_tensor(img):
        img = (img.cpu().numpy() * 255).astype(np.uint8)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(img)
        features = sam_model.model.image_encoder(torch.from_numpy(img).to(device))
    
    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).reshape(1, features.shape[1], -1)
    
    return torch.nn.functional.normalize(features, dim=1)