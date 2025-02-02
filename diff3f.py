import torch
from PIL import Image
import numpy as np
from diffusion import add_texture_to_render
from sam2_setup import get_sam_features
from dino import get_dino_features
from pytorch3d.ops import ball_query
from mesh_video_generator import MeshVideoGenerator
from tqdm import tqdm
from time import time
import random


FEATURE_DIMS_DIFFUSION = 1280
FEATURE_DIMS_DINO = 768
FEATURE_DIMS_SAM = 256
VERTEX_GPU_LIMIT = 35000


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
    sam_model,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
    use_sam=False,
    use_only_diffusion=False,
    use_diffusion=True,
    save_path=None
):
    """Main function to extract and map features to mesh vertices."""
    t1 = time()
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]

    # Calculate distances for ball query radius
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()

    ball_drop_radius = maximal_distance * tolerance

    # Initialize video generator and get renders
    video_gen = MeshVideoGenerator(hw=H, num_views=num_views, device=device)
    batched_renderings, normal_batched_renderings, camera, depth = video_gen.render_mesh_with_depth(mesh)
    
    if save_path:
        torch.save({
            'renderings': batched_renderings,
            'normal_renderings': normal_batched_renderings,
            'camera': camera,
            'depth': depth
        }, save_path)
        print(f"Rendered mesh saved to {save_path}")
   
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()

    os.makedirs("video", exist_ok=True)  
    video_gen.save_video(batched_renderings, f"video/{prompt}.mp4", fps=30, display_frames=False)
    
    # Setup pixel coordinates and grid
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()

    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feature_dims = 0
    if use_diffusion:
        feature_dims += FEATURE_DIMS_DIFFUSION
    if not use_only_diffusion:
        if use_sam:
            feature_dims += FEATURE_DIMS_SAM
        else:
            feature_dims += FEATURE_DIMS_DINO

    # Initialize feature storage
    ft_per_vertex = torch.zeros((len(mesh_vertices), feature_dims)).half()  # .to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)

    # Process each frame
    for idx in tqdm(range(len(batched_renderings))):
        # Calculate world coordinates for current frame
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device)
        
        # Process frame and map features
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)


        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image
        )


        ### TODO
        if not use_only_diffusion:
            if not use_sam:
                aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0], grid)
            else:
                aligned_dino_features = get_sam_features(device, sam_model, diffusion_input_img, grid)
        ###

        #aligned_dino_features = get_sam_features(device, sam_model, diffusion_input_img, grid)

        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
        # this is feature per pixel in the grid
        ### TODO
        if not use_only_diffusion:
            if not use_diffusion:
                aligned_features = aligned_dino_features
            else:
                aligned_features = torch.hstack([aligned_features*0.5, aligned_dino_features*0.5])
        ###
        
        # aligned_features = aligned_dino_features
        features_per_pixel = aligned_features[0, :, indices].cpu()

        # map pixel to vertex on mesh
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print("Number of missing features: ", missing_features)
    print("Copied features from nearest vertices")

    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][
            closest_vertex_indices, :
        ]
    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    return ft_per_vertex