from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from normal_shading import HardPhongNormalShader
import torch
import math

@torch.no_grad()
def run_rendering_elevation_fix(
    device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False, fixed_angle=None
):
    # Calculate bounding box and center
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Calculate viewing distance based on object size
    bb_diff = bbox_max - bbox_min
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * scaling_factor

    # Calculate angles based on fixed_angle parameter
    angle_step = 360.0 / num_views
    azimuth = torch.linspace(0, 360 - angle_step, num_views)
    elevation = torch.full_like(azimuth, fixed_angle['value'] if fixed_angle else 0.0)

    # Move everything to device
    azimuth = azimuth.to(device)
    elevation = elevation.to(device)

    # Prepare camera transform
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance,
        azim=azimuth,
        elev=elevation,
        device=device,
        at=bbox_center
    )

    # Setup camera
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # Setup lighting
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device
    )

    # Setup renderer
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render from multiple views
    batch_mesh = mesh.extend(num_views)
    renderings = renderer(batch_mesh)

    # Handle normal map rendering if requested
    normal_renderings = None
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_renderings = normal_renderer(batch_mesh)

    # Get depth information
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf

    return renderings, normal_renderings, camera, depth

@torch.no_grad()
def run_rendering(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False, fixed_angle=None):
    """
    Render object with vertical rotation (elevation) while keeping azimuth fixed
    """
    # Calculate bounding box and center
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Calculate viewing distance based on object size
    bb_diff = bbox_max - bbox_min
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * scaling_factor

    # Fixed azimuth angle (in degrees)
    fixed_azimuth = float(fixed_angle['value']) if fixed_angle else 0.0
    
    # Create elevation sequence from 0 to 360 degrees
    elevation = torch.linspace(0, 360, num_views + 1)[:-1].to(device)
    
    # Convert angles to radians and ensure they're tensors on the correct device
    elevation_rad = elevation * (math.pi / 180.0)
    azimuth_rad = torch.tensor(fixed_azimuth * (math.pi / 180.0), device=device)
    
    # Calculate camera positions using spherical coordinates
    x = distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    y = distance * torch.sin(elevation_rad)
    z = distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    
    # Stack coordinates
    camera_positions = torch.stack([x, y, z], dim=-1)
    bbox_center = bbox_center.unsqueeze(0)

    # Calculate the up vector dynamically
    # Base up vector
    base_up = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    
    # Adjust up vector based on elevation angle
    up_vectors = []
    rot_adjusts = []
    
    for i in range(num_views):
        current_elevation = elevation[i].item()
        
        # Create rotation adjustment matrix for each view
        rot_adjust = torch.eye(3, device=device)
        
        # Handle the specific case around 90 degrees
        if 89 < current_elevation <= 95:
            up_vectors.append(-base_up)  # Invert up vector
            rot_adjust[0, 0] = 1  # Flip X axis
        elif 90 < current_elevation < 270:
            up_vectors.append(-base_up)
        else:
            up_vectors.append(base_up)
            
        # Add extra rotation for consistent orientation
        if current_elevation > 180:
            rot_adjust[0, 0] = 1
            
        rot_adjusts.append(rot_adjust.unsqueeze(0))
    
    up = torch.cat(up_vectors, dim=0)
    rot_adjust = torch.cat(rot_adjusts, dim=0)

    # Calculate look_at transform with adjusted up vectors
    rotation, translation = look_at_view_transform(
        eye=camera_positions,
        at=bbox_center.expand(num_views, -1),
        up=up,
        device=device
    )

    # Apply rotation adjustments
    rotation = torch.bmm(rotation, rot_adjust)

    # Setup camera
    camera = PerspectiveCameras(
        R=rotation,
        T=translation,
        device=device
    )

    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # Setup lighting with adjusted positions
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_positions,
        device=device
    )

    # Setup renderer
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render from multiple views
    batch_mesh = mesh.extend(num_views)
    renderings = renderer(batch_mesh)

    # Handle normal map rendering if requested
    normal_renderings = None
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_renderings = normal_renderer(batch_mesh)

    # Get depth information
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf

    return renderings, normal_renderings, camera, depth

def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False, fixed_angle=None):
    """
    Wrapper function to handle rendering with error handling
    """
    try:
        return run_rendering(device, mesh, mesh_vertices, num_views, H, W,
                           use_normal_map=use_normal_map, fixed_angle=fixed_angle)
    except torch.linalg.LinAlgError as e:
        print("Linear algebra error during rendering:", str(e))
        return None
    except Exception as e:
        print("Error during rendering:", str(e))
        return None
    