from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from normal_shading import HardPhongNormalShader
import torch
import math

@torch.no_grad()
def calculate_bbox_and_distance(mesh, scaling_factor):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    bb_diff = bbox_max - bbox_min
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * scaling_factor
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
    angle_step = 360.0 / num_views
    azimuth = torch.linspace(0, 360 - angle_step, num_views).to(device)
    elevation = torch.full_like(azimuth, fixed_angle['value'] if fixed_angle else 0.0).to(device)

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
    fixed_azimuth = float(fixed_angle['value']) if fixed_angle else 0.0
    elevation = torch.linspace(0, 360, num_views + 1)[:-1].to(device)

    elevation_rad = elevation * (math.pi / 180.0)
    azimuth_rad = torch.tensor(fixed_azimuth * (math.pi / 180.0), device=device)

    x = distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    y = distance * torch.sin(elevation_rad)
    z = distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)

    camera_positions = torch.stack([x, y, z], dim=-1)
    bbox_center = bbox_center.unsqueeze(0)

    base_up = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    up_vectors = []
    rot_adjusts = []

    elevations = elevation.cpu().numpy()  # Convert to numpy for comparison
    for i in range(num_views):
        current_elevation = elevations[i]
        rot_adjust = torch.eye(3, device=device)

        if 89 < current_elevation <= 95:
            up_vectors.append(-base_up)
            rot_adjust[0, 0] = 1
        elif 90 < current_elevation < 270:
            up_vectors.append(-base_up)
        else:
            up_vectors.append(base_up)

        if current_elevation > 180:
            rot_adjust[0, 0] = 1

        rot_adjusts.append(rot_adjust.unsqueeze(0))

    up = torch.cat(up_vectors, dim=0)
    rot_adjust = torch.cat(rot_adjusts, dim=0)

    rotation, translation = look_at_view_transform(
        eye=camera_positions,
        at=bbox_center.expand(num_views, -1),
        up=up,
        device=device
    )
    rotation = torch.bmm(rotation, rot_adjust)

    return rotation, translation, camera_positions

@torch.no_grad()
def run_rendering(device, mesh, num_views, H, W, use_normal_map=False, fixed_angle=None):
    scaling_factor = 0.65
    bbox_center, distance = calculate_bbox_and_distance(mesh, scaling_factor)

    if fixed_angle and fixed_angle['type'] == 'elevation':
        rotation, translation = compute_camera_transform_fixed_elevation(
            device, num_views, bbox_center, distance, fixed_angle
        )
        camera_positions = None
    elif fixed_angle and fixed_angle['type'] == 'azimuth':
        rotation, translation, camera_positions = compute_camera_transform_fixed_azimuth(
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

    normal_renderings = None
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_renderings = normal_renderer(batch_mesh)

    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf

    return renderings, normal_renderings, camera, depth

def batch_render(device, mesh, num_views, H, W, use_normal_map=False, fixed_angle=None):
    try:
        print(f"Starting batch_render with num_views={num_views}, H={H}, W={W}")
        print(f"Fixed angle settings: {fixed_angle}")
        print(f"Device: {device}")
        print(f"Mesh info: vertices={mesh.verts_packed().shape}, faces={mesh.faces_packed().shape}")
        
        result = run_rendering(device, mesh, num_views, H, W, use_normal_map, fixed_angle)
        print("Rendering completed successfully")
        return result
    except torch.linalg.LinAlgError as e:
        print(f"Linear algebra error during rendering: {str(e)}")
        print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        return None
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None