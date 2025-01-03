from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from normal_shading import HardPhongNormalShader
import torch
import math

@torch.no_grad()
def calculate_bbox_and_distance(mesh, scaling_factor=2.0):
    """
    Calculate bounding box center and appropriate camera distance.
    Added safeguards for distance calculation.
    """
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    # Calculate diagonal length of bounding box
    bb_diff = bbox_max - bbox_min
    diagonal = torch.sqrt((bb_diff * bb_diff).sum())
    
    # Set distance based on diagonal length and scaling factor
    # This ensures the object will be fully visible
    distance = diagonal * scaling_factor
    
    # Add minimum distance threshold
    min_distance = 1.0  # Adjust this value if needed
    distance = torch.max(distance, torch.tensor(min_distance, device=distance.device))
    
    return bbox_center, distance

@torch.no_grad()
def setup_rasterizer_and_lights(device, camera, image_size, camera_positions=None):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    light_location = camera_positions if camera_positions is not None else camera.get_camera_center()
    
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=light_location,
        device=device
    )

    return rasterizer, lights

@torch.no_grad()
def compute_camera_transform_fixed_elevation(device, num_views, bbox_center, distance, fixed_angle):
    """ angle_step = 360.0 / num_views
    azimuth = torch.linspace(0, 360 - angle_step, num_views).to(device)
    elevation = torch.full_like(azimuth, fixed_angle['value']).to(device)
    #elevation = torch.full_like(azimuth, fixed_angle['value'] if fixed_angle else 0.0).to(device)
 """
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elevation = torch.linspace(start=0, end=end, steps=steps).repeat(steps)
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps)

    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance,
        azim=azimuth,
        elev=elevation,
        device=device,
        at=bbox_center
    )
    return rotation, translation

@torch.no_grad()
def compute_camera_transform_fixed_azimuth(device, num_views, bbox_center, distance, fixed_angle):
    """
    Compute camera transforms for a vertical circular path around an object,
    with improved stability for large number of views.
    """
    fixed_azimuth = float(fixed_angle['value']) if fixed_angle else 0.0
    
    # Generate elevation angles (going from 0 to 360 degrees)
    # Add small epsilon to avoid exact 90/270 degree angles which can cause singularities
    epsilon = 0.001
    elevation = torch.linspace(epsilon, 360 - epsilon, num_views).to(device)
    elevation_rad = elevation * (math.pi / 180.0)
    azimuth_rad = torch.tensor(fixed_azimuth * (math.pi / 180.0), device=device)

    # Calculate normalized camera positions on a unit sphere
    x = torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    y = torch.sin(elevation_rad)
    z = torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    
    # Stack and normalize camera positions
    camera_directions = torch.stack([x, y, z], dim=-1)
    camera_directions = camera_directions / (torch.norm(camera_directions, dim=-1, keepdim=True) + 1e-8)
    
    # Scale by distance to get actual camera positions
    camera_positions = bbox_center.unsqueeze(0) + (camera_directions * distance)
    
    up_vectors = []
    flip_flags = []
    
    # Calculate view directions with added stability
    view_dirs = bbox_center.unsqueeze(0) - camera_positions
    view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
    
    for i in range(num_views):
        current_elevation = elevation[i].item()
        view_dir = view_dirs[i]
        
        # Avoid exact vertical orientations
        if abs(current_elevation - 90) < 1 or abs(current_elevation - 270) < 1:
            current_elevation += epsilon
            
        if current_elevation < 90 or current_elevation >= 270:
            up = torch.tensor([0., 1., 0.], device=device)
            flip_flags.append(False)
        else:
            up = torch.tensor([0., 1., 0.], device=device)
            flip_flags.append(True)
        
        # Add stability to cross products
        right = torch.cross(view_dir, up, dim=0)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(right, view_dir, dim=0)
        up = up / (torch.norm(up) + 1e-8)
        
        up_vectors.append(up.unsqueeze(0))
    
    up = torch.cat(up_vectors, dim=0)
    
    # Add checks for invalid transforms
    if torch.any(torch.isnan(camera_positions)) or torch.any(torch.isnan(up)):
        raise ValueError("Invalid camera transforms detected")
        
    rotation, translation = look_at_view_transform(
        eye=camera_positions,
        at=bbox_center.expand(num_views, -1),
        up=up,
        device=device
    )
    
    return rotation, translation, camera_positions, flip_flags


@torch.no_grad()
def run_rendering(device, mesh, num_views, H, W, use_normal_map=False, fixed_angle=None):
    scaling_factor = 0.8
    bbox_center, distance = calculate_bbox_and_distance(mesh, scaling_factor)

    if fixed_angle and fixed_angle['type'] == 'elevation':
        rotation, translation = compute_camera_transform_fixed_elevation(
            device, num_views, bbox_center, distance, fixed_angle
        )
        camera_positions = None
        flip_flags = None
    elif fixed_angle and fixed_angle['type'] == 'azimuth':
        rotation, translation, camera_positions, flip_flags = compute_camera_transform_fixed_azimuth(
            device, num_views, bbox_center, distance, fixed_angle
        )
    else:
        raise ValueError("fixed_angle must specify 'type' as 'elevation' or 'azimuth'.")

    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterizer, lights = setup_rasterizer_and_lights(device, camera, (H, W), camera_positions)

    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    batch_mesh = mesh.extend(num_views)
    renderings = renderer(batch_mesh)

    # Apply flipping to maintain visual consistency
    if flip_flags is not None:
        for i, should_flip in enumerate(flip_flags):
            if should_flip:
                # Flip the image vertically and horizontally
                renderings[i] = torch.flip(torch.flip(renderings[i], [0]), [1])

    normal_renderings = None
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_renderings = normal_renderer(batch_mesh)
        
        # Apply same flipping to normal map
        if flip_flags is not None:
            for i, should_flip in enumerate(flip_flags):
                if should_flip:
                    normal_renderings[i] = torch.flip(torch.flip(normal_renderings[i], [0]), [1])

    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf

    return renderings, normal_renderings, camera, depth

def batch_render(device, mesh, num_views, H, W, use_normal_map=False, fixed_angle=None):
    """
    Wrapper function with additional error checking and debugging
    """
    try:
        print(f"Starting batch_render with num_views={num_views}, H={H}, W={W}")
        
        # Validate mesh
        if mesh.verts_packed().shape[0] == 0:
            raise ValueError("Mesh has no vertices")
            
        # Add validation for mesh scale and center
        verts = mesh.verts_packed()
        if torch.any(torch.isnan(verts)):
            raise ValueError("Mesh contains NaN vertices")
            
        # Check for degenerate mesh
        if torch.any(torch.std(verts, dim=0) < 1e-6):
            raise ValueError("Mesh appears to be degenerate (flat in one or more dimensions)")
            
        # Run rendering with validated parameters
        result = run_rendering(device, mesh, num_views, H, W, use_normal_map, fixed_angle)
        
        # Validate results
        if result[0] is None or result[1] is None:
            raise ValueError("Rendering produced null results")
            
        print("Rendering completed successfully")
        return result
        
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None