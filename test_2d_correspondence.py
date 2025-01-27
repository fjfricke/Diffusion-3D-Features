import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
from sam2_setup import init_sam2, get_sam_features, run_pca_on_specific_embeddings
from dino import get_dino_features, init_dino
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_images(img_path_1, img_path_2):
    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)
    # convert to rgb
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    return img_1, img_2

def resize_features_to_image_size(features, original_image_size):
    # Assuming features is a tensor of shape (1, 256, 64, 64)
    # and original_image_size is a tuple (height, width) e.g., (360, 540)
    _, _, h, w = features.shape
    target_height, target_width = original_image_size

    # Use bilinear interpolation to resize
    resized_features = F.interpolate(features, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return resized_features


def calculate_correspondence_in_img2_for_a_point_in_img1(point_in_feat_img1, feat_img_1, feat_img_2):
    # Extract the feature vector for the given point in img1
    feature_vector_img1 = feat_img_1[:, :, point_in_feat_img1[1], point_in_feat_img1[0]]
    print(feature_vector_img1.shape)
    # Calculate the Euclidean distance between the feature vector of img1 and all feature vectors in img2
    distances = torch.norm(feat_img_2 - feature_vector_img1.unsqueeze(-1).unsqueeze(-1), dim=1)

    # Normalize the distances to create a heatmap
    heatmap = (distances - distances.min()) / (distances.max() - distances.min())

    return heatmap

def plot_heatmap(original_img, heatmap, display=True, save_to=False):
    # Convert the heatmap to a numpy array and resize it to match the original image size
    heatmap_resized = heatmap.squeeze(0).cpu().numpy()
    # heatmap_resized = cv2.resize(heatmap_np, (original_img.shape[1], original_img.shape[0]))

    # Normalize the heatmap to be in the range [0, 1]
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    # Create a color map
    colormap = plt.get_cmap('jet')

    # Apply the colormap to the heatmap
    heatmap_colored = colormap(heatmap_resized)

    # Overlay the heatmap on the original image
    overlayed_img = (0.5 * original_img + 0.5 * heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Display the image
    if display:
        plt.imshow(overlayed_img)
        plt.axis('off')
        plt.show()
        plt.imshow(heatmap_resized, cmap='jet')
        plt.axis('off')
        plt.show()
    if save_to:
        Image.fromarray(overlayed_img).save(save_to)

def display_image_with_point(image, point, display=True, save_to=False):

    def draw_point_on_image(image, point, color=(0, 255, 0), radius=5, thickness=2):
        # Draw a circle on the image at the specified point
        image_with_point = cv2.circle(image.copy(), point, radius, color, thickness)
        return image_with_point

    # Draw the point on the image
    image_with_point = draw_point_on_image(image, point)

    # Display the image
    if display:
        plt.imshow(image_with_point)
        plt.axis('off')
        plt.show()
    if save_to:
        Image.fromarray(image_with_point).save(save_to)

def run_2d_correspondence(dino_or_sam="sam", display=True, save_to=False, layer_name="image_encoder.trunk.blocks.24", use_pca_with_n_components=False, point="ear"):
    img_1, img_2 = load_images("test_images/input_cow.jpg", "test_images/input_cow_2.jpg")
    # img_1 = cv2.resize(img_1, (1024, 1024))
    # img_2 = cv2.resize(img_2, (1024, 1024))
    device = "mps"
    if dino_or_sam == "sam":
        sam_model = init_sam2(device=device)
        features_1 = get_sam_features(device=device, sam_model=sam_model, img=img_1, get_features_directly=True, layer_name=layer_name)
        features_2 = get_sam_features(device=device, sam_model=sam_model, img=img_2, get_features_directly=True, layer_name=layer_name)
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
    else:
        img_1_pil = Image.fromarray(img_1)
        img_2_pil = Image.fromarray(img_2)
        dino_model = init_dino(device=device)
        features_1 = get_dino_features(device=device, dino_model=dino_model, img=img_1_pil, get_features_directly=True)
        features_2 = get_dino_features(device=device, dino_model=dino_model, img=img_2_pil, get_features_directly=True)
    features_1 = resize_features_to_image_size(features_1, img_1.shape[:2])
    features_2 = resize_features_to_image_size(features_2, img_2.shape[:2])
    if save_to:
        save_to = f"{save_to}_pca_img1.png"
    pca_features_1 = run_pca_on_specific_embeddings(features_1, display=display, save_to=save_to, n_components=use_pca_with_n_components)
    if save_to:
        save_to = f"{save_to}_pca_img2.png"
    pca_features_2 = run_pca_on_specific_embeddings(features_2, display=display, save_to=save_to, n_components=use_pca_with_n_components)

    if use_pca_with_n_components:
        features_1 = pca_features_1
        features_2 = pca_features_2

    if point == "ear":
        point_in_feat_img1 = (65, 58)
    elif point == "leg":
        point_in_feat_img1 = (435, 321)
    else:
        raise ValueError(f"Point {point} not supported")
    if save_to:
        save_to = f"{save_to}_point_in_feat_img1.png"
    display_image_with_point(img_1, point_in_feat_img1, display=display, save_to=save_to)
    heatmap = calculate_correspondence_in_img2_for_a_point_in_img1(point_in_feat_img1, features_1, features_2)
    if save_to:
        save_to = f"{save_to}_heatmap.png"
    plot_heatmap(img_2, heatmap, display=display, save_to=save_to)

def create_box_mask_batches(img, batch_size=16):
    if img.shape[0] % batch_size != 0 or img.shape[1] % batch_size != 0:
        raise ValueError(f"Image dimensions {img.shape} are not divisible by batch size {batch_size}")
    # create list of mask batches where each is an ndarray with BHBH format marking the mask as a square ( e.g. [0, 0, 16, 16])
    mask_batches = []
    for i in range(0, img.shape[0], batch_size):
        for j in range(0, img.shape[1], batch_size):
            # Create a mask with BHBH format
            mask = np.array([i, j, i + batch_size, j + batch_size])
            mask_batches.append(mask)
    return mask_batches


def run_2d_correspondence_with_sam_masks(dino_or_sam="dino", display=True, save_to=False, layer_name="image_encoder.trunk.blocks.24", box_size=512):
    img_1, img_2 = load_images("test_images/input_cow.jpg", "test_images/input_cow_2.jpg")
    img_1 = cv2.resize(img_1, (1024, 1024))
    img_2 = cv2.resize(img_2, (1024, 1024))
    device = "mps"
    sam_model = init_sam2(device=device)
    box_mask_batches = create_box_mask_batches(img_1, box_size)
    features_1 = None
    features_2 = None
    for box_mask_batch in tqdm(box_mask_batches, desc="Processing mask batches"):
        mask_features_1 = get_sam_features(device=device, sam_model=sam_model, img=img_1, get_features_directly=True, layer_name=layer_name, box=box_mask_batch)
        mask_features_2 = get_sam_features(device=device, sam_model=sam_model, img=img_2, get_features_directly=True, layer_name=layer_name, box=box_mask_batch)
        mask_features_1 = resize_features_to_image_size(mask_features_1, img_1.shape[:2])
        mask_features_2 = resize_features_to_image_size(mask_features_2, img_2.shape[:2])
        if features_1 is None:
            features_1 = torch.zeros(mask_features_1.shape, device=mask_features_1.device)
        if features_2 is None:
            features_2 = torch.zeros(mask_features_2.shape, device=mask_features_2.device)
        # Extract mask coordinates
        # x1, y1, x2, y2 = box_mask_batch
        # mask_features_1 = mask_features_1[:, :, x1:x2, y1:y2]
        # mask_features_2 = mask_features_2[:, :, x1:x2, y1:y2]
        # mask_features_1 = F.normalize(mask_features_1, p=2, dim=1)
        # mask_features_2 = F.normalize(mask_features_2, p=2, dim=1)
        
        # Add features only within the masked region
        # features_1[:, :, x1:x2, y1:y2] = mask_features_1
        # features_2[:, :, x1:x2, y1:y2] = mask_features_2
        features_1 += mask_features_1
        features_2 += mask_features_2
    features_1 = F.normalize(features_1, p=2, dim=1)
    features_2 = F.normalize(features_2, p=2, dim=1)
    if save_to:
        save_to = f"{save_to}_pca_img1.png"
    run_pca_on_specific_embeddings(features_1, display=display, save_to=save_to)
    if save_to:
        save_to = f"{save_to}_pca_img2.png"
    run_pca_on_specific_embeddings(features_2, display=display, save_to=save_to)
    point_in_feat_img1 = (124, 168)
    if save_to:
        save_to = f"{save_to}_point_in_feat_img1.png"
    display_image_with_point(img_1, point_in_feat_img1, display=display, save_to=save_to)
    heatmap = calculate_correspondence_in_img2_for_a_point_in_img1(point_in_feat_img1, features_1, features_2)
    if save_to:
        save_to = f"{save_to}_heatmap.png"
    plot_heatmap(img_2, heatmap, display=display, save_to=save_to)


if __name__ == "__main__":
    # run_2d_correspondence_with_sam_masks(dino_or_sam="sam", display=True, save_to="test_images/output_batched/size_128_upscaling_0_full_add", layer_name="sam_mask_decoder.output_upscaling.0", box_size=128)
    run_2d_correspondence(dino_or_sam="sam", display=False, save_to="test_images/presentation/sam_last_upscale0/sam_last_upscale0_pca3_leg", layer_name="sam_mask_decoder.output_upscaling.0", use_pca_with_n_components=3, point="leg")
    # for i in range(48):
    #     print(f"Running layer {i}")
    #     run_2d_correspondence(dino_or_sam="sam", display=False, save_to=f"test_images/output_attn/sam_2d_correspondence_layer_{i}", layer_name=f"image_encoder.trunk.blocks.{i}.attn")


# "test_images/presentation/sam_last_upscale0/sam_last_upscale0_pca3"