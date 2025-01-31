import os
import cv2
import numpy as np
import torch
from render import batch_render
from utils import convert_mesh_container_to_torch_mesh
from dataloaders.mesh_container import MeshContainer
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.io import load_objs_as_meshes


class MeshVideoGenerator:
    def __init__(self, output_dir="outputs", hw=128, num_views=10, use_normal_map=True, device="cuda"):
        self.output_dir = output_dir
        self.hw = hw
        self.num_views = num_views
        self.use_normal_map = use_normal_map
        self.device = device
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_mesh_by_index(self, folder, index):
        # Get all files in the folder
        files = sorted(os.listdir(folder))
        # Filter only the files you want
        valid_extensions = {".obj", ".off", ".ply"}
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Check if the index is valid
        if index < 0 or index >= len(files):
            raise IndexError(f"Index out of range. Valid range: 0-{len(files) - 1}")
            
        # Get the corresponding file
        file_to_load = files[index]
        file_path = os.path.join(folder, file_to_load)
        print(f"Loading file: {file_path}")
        
        # Load the mesh
        mesh = load_objs_as_meshes([file_path], device=self.device)
        return mesh, file_to_load
    
    def render_mesh_with_depth(self, mesh):
        #torch_mesh = convert_mesh_container_to_torch_mesh(mesh, device=self.device, is_tosca=False)
        torch_mesh = mesh
        
        # Get both rotations with depth
        #azimuth_data = batch_render(
        azimuth_renderings, azimuth_normals, azimuth_camera, azimuth_depth = batch_render(
            device=self.device,
            mesh=torch_mesh,
            num_views=self.num_views,
            H=self.hw,
            W=self.hw,
            use_normal_map=self.use_normal_map,
            fixed_angle={'type': 'elevation', 'value': 0}
        )
        
        
        elevation_renderings, elevation_normals, elevation_camera, elevation_depth = batch_render(
            device=self.device,
            mesh=torch_mesh,
            num_views=self.num_views,
            H=self.hw,
            W=self.hw,
            use_normal_map=self.use_normal_map,
            fixed_angle={'type': 'azimuth', 'value': 0}
        )
        


        try:
            combined_renderings = torch.cat([azimuth_renderings, elevation_renderings], dim=0)
            combined_depth = torch.cat([azimuth_depth, elevation_depth], dim=0)
            
            if self.use_normal_map and elevation_normals is not None and azimuth_normals is not None:
                combined_normals = torch.cat([azimuth_normals, elevation_normals], dim=0)
            else:
                combined_normals = None
                
            # Concatenate cameras by combining their R and T matrices
            R = torch.cat([azimuth_camera.R, elevation_camera.R], dim=0)
            T = torch.cat([azimuth_camera.T, elevation_camera.T], dim=0)
            
            # Create new combined camera
            combined_camera = PerspectiveCameras(
                R=R,
                T=T,
                device=self.device
            )
            
            return combined_renderings, combined_normals, combined_camera, combined_depth
        
        except Exception as e:
            print(f"Error concatenating renderings: {str(e)}")
            print(f"Elevation renderings shape: {elevation_renderings.shape}")
            print(f"Azimuth renderings shape: {azimuth_renderings.shape}")
            return None, None, None, None
    
    def render_mesh(self, mesh):

        # Convert mesh to torch format
        #torch_mesh = convert_mesh_container_to_torch_mesh(mesh, device=self.device, is_tosca=False)
        torch_mesh = mesh
        # Render azimuth rotation
        azimuth_renders, normal_azimuth, camera, depth = batch_render(
            device=self.device,
            mesh=torch_mesh,
            num_views=self.num_views,
            H=self.hw,
            W=self.hw,
            use_normal_map=self.use_normal_map,
            fixed_angle={'type': 'elevation', 'value': 0}
        )
        
        # Render elevation rotation
        elevation_renders, normal_elevation, camera_elev, depth_elev = batch_render(
            device=self.device,
            mesh=torch_mesh,
            num_views=self.num_views,
            H=self.hw,
            W=self.hw,
            use_normal_map=self.use_normal_map,
            fixed_angle={'type': 'azimuth', 'value': 0}
        )
        
        # Combine azimuth and elevation renders
        combined_renders = torch.cat([azimuth_renders, elevation_renders], dim=0)
        return combined_renders

    def save_video(self, renderings, output_path, fps=30, display_frames=False):
        # Move to CPU first, then convert to numpy
        renderings = renderings.cpu().numpy()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.hw, self.hw))
        
        total_frames = len(renderings)
        for i, view in enumerate(renderings):
            frame = (view * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
            if display_frames:
                plt.figure(figsize=(8, 8))
                plt.imshow(view)
                if i < self.num_views:
                    plt.title(f"Azimuth: {(i/self.num_views)*360:.1f}°")
                else:
                    plt.title(f"Elevation: {((i-self.num_views)/self.num_views)*360:.1f}°")
                plt.axis("off")
                plt.show()
                plt.close()
            
        out.release()
        print(f"Video saved to {output_path}")
        print()

    def process_single_mesh(self, folder, index, display_frames=False):
        try:
            mesh, filename = self.load_mesh_by_index(folder, index)
            base_filename = os.path.splitext(filename)[0]
            
            # Get combined renders (azimuth + elevation)
            combined_renders = self.render_mesh(mesh)
            
            # Save combined video
            output_path = os.path.join(self.output_dir, f"{base_filename}_combined.mp4")
            self.save_video(combined_renders, output_path, display_frames=display_frames)
                
        except Exception as e:
            print(f"Error processing mesh {index}: {str(e)}")

    def process_folder(self, folder, display_frames=False):
        """Process all meshes in the given folder"""
        files = sorted(os.listdir(folder))
        valid_extensions = {".obj", ".off"}
        mesh_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"Found {len(mesh_files)} mesh files to process")
        
        for i in range(len(mesh_files)):
            print(f"\nProcessing mesh {i+1}/{len(mesh_files)}")
            self.process_single_mesh(folder, i, display_frames=display_frames)