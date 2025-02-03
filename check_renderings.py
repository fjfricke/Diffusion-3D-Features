from mesh_video_generator import MeshVideoGenerator
from utils import convert_mesh_container_to_torch_mesh, load_mesh

import sys
sys.path.append('/Users/felix/Programming/TUM/Diffusion-3D-Features/dino-tracker/adl4cv')

from load_files import compute_optical_flow_with_mask
from plot_flow import visualize_optical_flow_video


if __name__ == "__main__":
    video_generator = MeshVideoGenerator(hw=512, num_views=50, device='cpu')

    mesh = load_mesh("/Users/felix/Programming/TUM/Diffusion-3D-Features/meshes/cow.obj", device='cpu')

    mesh = (mesh if hasattr(mesh, 'verts_list') 
            else convert_mesh_container_to_torch_mesh(mesh, device='cpu', is_tosca=False))

    combined_renderings, combined_normals, combined_camera, combined_depth = video_generator.render_mesh_with_depth(mesh=mesh)

    flows, masks = compute_optical_flow_with_mask(combined_camera, combined_depth)

    visualize_optical_flow_video(flows, masks)
