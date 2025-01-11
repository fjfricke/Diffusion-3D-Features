import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from sam2.modeling.sam2_base import SAM2Base
from dino import init_dino, get_dino_features


def init_sam2(device, model_size="large"):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # model = model.to(device).eval()c\
    return model


torch.no_grad()
def get_sam_features(device, sam_model, img, grid=None, get_features_directly=False):
    intermediate_embeddings = []
    add_hook_to_get_embeddings_for_layers(sam_model, ["sam_mask_decoder.output_upscaling.3", "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.out_proj"], intermediate_embeddings)
   # Use SAM's predictor to get features
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(img)
    # Get features from the specified intermediate layer
    # features = predictor._features["image_embed"]
    predictor.predict()
    features = intermediate_embeddings[0]
    # features = predictor._features["high_res_feats"][0]

    if features.dim() == 3:
        B, HW, D = features.shape
        H, W = int(HW**0.5), int(HW**0.5)
        features = features.view(B, H, W, D).permute(0, 3, 1, 2)
    if get_features_directly:
        return features


    # Process features similar to DINO
    h, w = features.shape[2], features.shape[3]  # Get spatial dimensions directly from features
    dim = features.shape[1]  # Feature dimension is in channel position
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = features.half()

    if grid is not None:
        features = torch.nn.functional.grid_sample(
            features, grid, align_corners=False
        ).reshape(1, dim, -1)
        features = torch.nn.functional.normalize(features, dim=1)
        return features
    else:
        features = torch.nn.functional.normalize(features, dim=1)
        return features
    
def get_test_image():
    import cv2
    img = cv2.imread('test_images/input_cow.jpg')
    if img is None:
        raise Exception("Could not load test image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def add_hook_to_get_embeddings_for_layers(sam_model, layer_names, embeddings):
    """
    Add a hook to get embeddings for specific layers
    
    inputs:
        sam_model: the SAM2 model
        layer_ids: the ids of the layers to get embeddings for
        embeddings: the list to append the embeddings to
    """

    def hook_fn(module, input, output):
        embeddings.append(output)

    for name, layer in sam_model.named_modules():
        if name in layer_names:
            layer.register_forward_hook(hook_fn)

def get_information_on_intermediate_embeddings():
    device = "cuda"
    sam_model: SAM2Base = init_sam2(device)
    intermediate_embeddings = []  # Initialize as a list
    intermediate_embeddings_decoder = []

    def hook_fn(module, input, output, name):
        intermediate_embeddings.append((name, output))
    
    def hook_fn_decoder(module, input, output, name):
        intermediate_embeddings_decoder.append((name, output))
    
    for name, layer in sam_model.named_modules():
        layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
    
    for name, layer in sam_model.sam_mask_decoder.named_modules():
        layer.register_forward_hook(lambda module, input, output, name=name: hook_fn_decoder(module, input, output, name))
    
    img = get_test_image()
    features = get_sam_features(device, sam_model, img)

    # Create a table of intermediate embeddings
    print("\nIntermediate Embeddings:")
    print("-" * 80)
    print(f"{'Index':^10} | {'Layer Name':^30} | {'Output Shape':^35}")
    print("-" * 80)
    
    for idx, (name, output) in enumerate(intermediate_embeddings):
        if isinstance(output, torch.Tensor):
            shape_str = str(tuple(output.shape))
        else:
            shape_str = "Not a tensor"
        print(f"{idx:^10} | {name:^30} | {shape_str:^35}")
    print("-" * 80)

    print("\nIntermediate Embeddings Decoder:")
    print("-" * 80)
    print(f"{'Index':^10} | {'Layer Name':^30} | {'Output Shape':^35}")
    print("-" * 80)
    for idx, (name, output) in enumerate(intermediate_embeddings_decoder):
        if isinstance(output, torch.Tensor):
            shape_str = str(tuple(output.shape))
        else:
            shape_str = "Not a tensor"
        print(f"{idx:^10} | {name:^30} | {shape_str:^35}")
    print("-" * 80)

    # run_pca_on_specific_embeddings(intermediate_embeddings_decoder[60][1], 'sam_features_pca_decoder_60')
    return intermediate_embeddings, intermediate_embeddings_decoder

def run_pca_on_specific_embeddings(embeddings, img_name=False):
    from sklearn.decomposition import PCA
    from PIL import Image
    # Reshape features for PCA
    features_reshaped = embeddings.squeeze().permute(1, 2, 0).reshape(-1, embeddings.shape[1]).cpu().numpy()

    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)

    # Normalize the PCA features to 0-255 for RGB visualization
    features_pca -= features_pca.min(axis=0)
    features_pca /= features_pca.ptp(axis=0)  # Use peak-to-peak (max-min) for normalization
    features_pca *= 255.0
    features_pca = features_pca.astype(np.uint8)

    # Reshape back to image dimensions
    h, w = embeddings.shape[2], embeddings.shape[3]
    feature_img = features_pca.reshape(h, w, 3)

    if img_name:
        Image.fromarray(feature_img).save(f'test_images/{img_name}.png')

    return feature_img


def run_test_pca():

    import cv2
    from PIL import Image
    from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt

    device = "cuda"
    sam_model = init_sam2(device)
    dino_model = init_dino(device)

    # # Load first frame from video
    # cap = cv2.VideoCapture("SHREC19_videos/1_combined.mp4")
    # ret, frame = cap.read()
    # cap.release()
    
    # if not ret:
    #     raise Exception("Could not load video frame")
    
    # # Save frame as image
    # cv2.imwrite('input_frame.png', frame)
        
    # # Convert BGR to RGB
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load test image
    img = cv2.imread('test_images/input_cow.jpg')
    if img is None:
        raise Exception("Could not load test image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features = get_sam_features(device, sam_model, img, get_features_directly=True)
    features_dino = get_dino_features(device, dino_model, img, grid=None)

    # Reshape features for PCA
    features_reshaped = features.squeeze().permute(1, 2, 0).reshape(-1, features.shape[1]).cpu().numpy()

    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)

    # Normalize the PCA features to 0-255 for RGB visualization
    features_pca -= features_pca.min(axis=0)
    features_pca /= features_pca.ptp(axis=0)  # Use peak-to-peak (max-min) for normalization
    features_pca *= 255.0
    features_pca = features_pca.astype(np.uint8)

    # Reshape back to image dimensions
    h, w = features.shape[2], features.shape[3]
    feature_img = features_pca.reshape(h, w, 3)

    # Save visualization
    Image.fromarray(feature_img).save('test_images/sam_features_viz.png')

if __name__ == "__main__":
    # get_information_on_intermediate_embeddings()
    run_test_pca()