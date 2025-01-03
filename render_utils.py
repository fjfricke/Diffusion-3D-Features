import torch
from PIL import Image
import numpy as np
from dataloaders.mesh_container import MeshContainer
from render import batch_render
from diff3f import arange_pixels
from tqdm import tqdm
from time import time
import random
import os
from pathlib import Path

from utils import convert_mesh_container_to_torch_mesh

VERTEX_GPU_LIMIT = 35000

def  render_and_save_mesh(
    device,
    mesh,
    output_dir,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_normal_map=True,
):
    """
    Render mesh from multiple viewpoints and save the rendered images.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        t1 = time()
        
        mesh_vertices = mesh.verts_list()[0]
        if len(mesh_vertices) > VERTEX_GPU_LIMIT:
            samples = random.sample(range(len(mesh_vertices)), 10000)
            maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
        else:
            maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()
        
        ball_drop_radius = maximal_distance * tolerance
        
        batched_renderings, normal_batched_renderings, camera, depth = batch_render(
            device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map
        )
        print("Rendering complete")
        
        if use_normal_map:
            normal_batched_renderings = normal_batched_renderings.cpu()
        batched_renderings = batched_renderings.cpu()
        
        pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
        pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
        camera = camera.cpu()
        depth = depth.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save rendered images
        for idx in tqdm(range(len(batched_renderings)), desc="Saving renders"):
            # Save regular render
            render_img = (batched_renderings[idx, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(render_img)
            img.save(os.path.join(output_dir, f'render_view_{idx:03d}.png'))

            # Save normal map if requested
            if use_normal_map:
                # Get the normal map and properly reshape it
                normal_img = normal_batched_renderings[idx].cpu().numpy()         
                       
                # Ensure correct shape (H, W, 3)
                if len(normal_img.shape) > 3:
                    normal_img = normal_img.squeeze()  # Remove any extra dimensions
                
                # If the shape is still wrong, try to reshape it
                if normal_img.shape != (H, W, 3):
                    normal_img = normal_img.reshape(H, W, 3)
                
                normal_img = (normal_img * 255).astype(np.uint8)
                normal_img = Image.fromarray(normal_img, mode='RGB')
                normal_img.save(os.path.join(output_dir, f'normal_map_view_{idx:03d}.png'))

            # Save depth map
            depth_img = depth[idx, :, :, 0].cpu().numpy()
            valid_depth = depth_img[depth_img != -1]
            if len(valid_depth) > 0:
                depth_img = np.where(depth_img != -1,
                                   ((depth_img - valid_depth.min()) / (valid_depth.max() - valid_depth.min()) * 255),
                                   0)
            depth_img = depth_img.astype(np.uint8)
            depth_img = Image.fromarray(depth_img, mode='L')  # Use 'L' mode for grayscale
            depth_img.save(os.path.join(output_dir, f'depth_view_{idx:03d}.png'))

        print(f"Rendering and saving completed in {time() - t1:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        print(f"Current array shape: {normal_img.shape if 'normal_img' in locals() else 'unknown'}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    source_file_path = Path(__file__).parent / "meshes/cow.obj"
    source_mesh = MeshContainer().load_from_file(source_file_path)
    num_views = 100
    H = 512
    W = 512
    tolerance = 0.004
    use_normal_map = True
    device = torch.device('cpu')

    mesh = convert_mesh_container_to_torch_mesh(source_mesh, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    render_and_save_mesh(
        device=device,
        mesh=mesh,
        output_dir=Path(__file__).parent / "meshes/cow_img",
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        use_normal_map=use_normal_map,
    )