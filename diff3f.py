import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from diffusion import add_texture_to_render
from dino import get_dino_features
from mesh_video_generator import MeshVideoGenerator
from render import batch_render
from pytorch3d.ops import ball_query
from tqdm import tqdm
from time import time
import random


FEATURE_DIMS = 1280+768 # diffusion unet + dino
VERTEX_GPU_LIMIT = 35000
SAM_FEATURE_DIMS = 256  # SAM's default feature dimension


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    device,
    sam_model,  # SAM2 model instead of pipe and dino
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    mesh_vertices=None,
    return_image=True,
    bq=True,
):
    t1 = time()
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    
    maximal_distance = calculate_max_distance(mesh_vertices)
    ball_drop_radius = maximal_distance * tolerance
    
    # Get both azimuth and elevation renders with depth
    video_gen = MeshVideoGenerator(hw=H, num_views=num_views//2, device=device)
    renders, depths, cameras = video_gen.render_mesh_with_depth(mesh)
    
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    
    ft_per_vertex = torch.zeros((len(mesh_vertices), SAM_FEATURE_DIMS)).half()
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()
    
    # Process video frames with SAM2
    for idx in tqdm(range(len(renders))):
        dp = depths[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        
        world_coords = cameras[idx].unproject_points(
            xy_depth, world_coordinates=True, from_ndc=True
        ).to(device)
        
        # Get SAM2 features for current frame
        frame = (renders[idx].cpu().numpy() * 255).astype(np.uint8)
        sam_features = sam_model.get_image_embedding(frame)
        
        # Map features to vertices
        if bq:
            queried_indices = ball_query(
                world_coords.unsqueeze(0),
                mesh_vertices.unsqueeze(0),
                K=100,
                radius=ball_drop_radius,
                return_nn=False
            ).idx[0].cpu()
            
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += sam_features.repeat_interleave(repeat, dim=1).T
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += sam_features.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    # Average and fill missing features
    average_and_fill_features(ft_per_vertex, ft_per_vertex_count, mesh_vertices)
    
    print(f"Time taken in mins: {(time() - t1) / 60}")
    return ft_per_vertex


def calculate_max_distance(mesh_vertices):
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        return torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    return torch.cdist(mesh_vertices, mesh_vertices).max()


def average_and_fill_features(ft_per_vertex, ft_per_vertex_count, mesh_vertices):
    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print(f"Number of missing features: {missing_features}")
    
    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2)
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][closest_vertex_indices, :]
