@torch.no_grad()
def run_rendering(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False, fixed_angle=None):
    """
    Render object from multiple viewpoints in a continuous sequence
    Args:
        ...existing args...
        fixed_angle: dict with 'type' ('azimuth' or 'elevation') and 'value' (angle in degrees)
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
    
    # Calculate angles based on fixed_angle parameter
    angle_step = 360.0 / num_views
    
    if fixed_angle is None or fixed_angle['type'] == 'elevation':
        # Vary azimuth, fix elevation
        azimuth = torch.linspace(0, 360 - angle_step, num_views)
        elevation = torch.full_like(azimuth, fixed_angle['value'] if fixed_angle else 0.0)
    else:  # fixed_angle['type'] == 'azimuth'
        # Create a continuous elevation path that goes up one side and down the other
        fixed_azi = float(fixed_angle['value'])
        half_views = num_views // 2
        
        # Generate elevation angles: 0 -> 180 -> 0
        elevation_up = torch.linspace(0, 180, half_views)
        elevation_down = torch.linspace(180, 0, num_views - half_views)
        elevation = torch.cat([elevation_up, elevation_down[1:]])  # Remove duplicate at 180
        
        # Create azimuth tensor with same size as elevation
        azimuth = torch.ones(num_views, device=device) * fixed_azi
        # Flip azimuth for second half
        azimuth[half_views:] += 180.0
    
    # Move everything to device and ensure shapes are correct
    azimuth = azimuth.to(device)
    elevation = elevation.to(device)
    distances = torch.full_like(azimuth, distance)
    
    # Ensure bbox_center is correctly broadcast for each view
    bbox_center = bbox_center.unsqueeze(0).expand(num_views, -1)
    
    # Prepare camera transform
    rotation, translation = look_at_view_transform(
        dist=distances,  # Now using per-view distances
        azim=azimuth,
        elev=elevation,
        device=device,
        at=bbox_center  # Now properly broadcast
    )

    # Setup camera with batch size
    cameras = PerspectiveCameras(
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
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # Setup lighting - ensure it's properly batched
    camera_center = cameras.get_camera_center()  # This should now be [num_views, 3]
    lights = PointLights(
        diffuse_color=torch.full((num_views, 3), 0.4, device=device),
        ambient_color=torch.full((num_views, 3), 0.6, device=device),
        specular_color=torch.full((num_views, 3), 0.01, device=device),
        location=camera_center,
        device=device
    )
    
    # Setup renderer
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    
    # Render from multiple views
    batch_mesh = mesh.extend(num_views)
    renderings = renderer(batch_mesh)
    
    # Handle normal map rendering if requested
    normal_renderings = None
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=cameras, lights=lights)
        normal_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_renderings = normal_renderer(batch_mesh)
    
    # Get depth information
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    
    return renderings, normal_renderings, cameras, depth