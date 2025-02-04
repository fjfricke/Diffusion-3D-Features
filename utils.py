import gc
import torch
import numbers
import numpy as np
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
import meshplot as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
# from diff3f import get_features_per_vertex
from diff3f_dinotracker import get_features_per_vertex as get_features_per_vertex_dinotracker
from pytorch3d.io import load_objs_as_meshes
from dataloaders.mesh_container import MeshContainer
import os


def compute_features(
    device,
    sam_model,
    dino_model,
    pipe,
    mesh_input,
    prompt,
    num_views,
    H,
    W,
    tolerance,
    save_path=None,
    use_normal_map=True,
    tex=False,
    tex_mesh=None,
    num_images_per_prompt=1,
    bq=True,
    use_sam=False,
    use_only_diffusion=False,
    use_diffusion=True,
    is_tosca=False,
):
    # Convert to PyTorch3D mesh if needed
    mesh = (mesh_input if hasattr(mesh_input, 'verts_list') 
            else convert_mesh_container_to_torch_mesh(mesh_input, device=device, is_tosca=is_tosca))
    if tex:
        tex_mesh = (tex_mesh if hasattr(tex_mesh, 'verts_list') 
                else convert_mesh_container_to_torch_mesh(tex_mesh, device=device, is_tosca=is_tosca))
    
    # Get mesh vertices
    mesh_vertices = mesh.verts_list()[0]

    # Compute features using all available models
    features = get_features_per_vertex(
        device=device,
        sam_model=sam_model,
        pipe=pipe, 
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        use_normal_map=use_normal_map,
        num_images_per_prompt=num_images_per_prompt,
        mesh_vertices=mesh_vertices,
        bq=bq,
        use_sam=use_sam,
        use_only_diffusion=use_only_diffusion,
        use_diffusion=use_diffusion,
        save_path=save_path,
        tex=tex,
        tex_mesh=tex_mesh
    )
    
    return features.cpu()

def compute_features_dinotracker(
    device,
    dinotracker_path,
    mesh_input,
    prompt,
    num_views,
    H,
    W,
    tolerance,
    use_normal_map=True,
    bq=True,
    is_tex=False
):
    # Convert to PyTorch3D mesh if needed
    mesh = (mesh_input if hasattr(mesh_input, 'verts_list') 
            else convert_mesh_container_to_torch_mesh(mesh_input, device=device, is_tosca=False))
    
    # Get mesh vertices
    mesh_vertices = mesh.verts_list()[0]

    # Compute features using all available models
    with torch.no_grad():
        features = get_features_per_vertex_dinotracker(
            device=device,
            dinotracker_path=dinotracker_path,
            mesh=mesh,
            prompt=prompt,
            num_views=num_views,
            H=H,
            W=W,
            tolerance=tolerance,
            use_normal_map=use_normal_map,
            mesh_vertices=mesh_vertices,
            bq=bq,
            is_tex=is_tex
        ).cpu()
    
    torch.cuda.empty_cache()
    gc.collect()
    return features

def generate_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 1
    value = 1
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors

def plot_mesh(myMesh,cmap=None):
    mp.plot(myMesh.vert, myMesh.face,c=cmap)

def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None, save_path='plot.html', show=False):
    from meshplot import website, jupyter
    import meshplot as mp
    import os
    
    # Get vertices and faces from PyTorch3D Meshes if needed
    if hasattr(myMesh1, 'verts_list'):
        verts1 = myMesh1.verts_list()[0].cpu().numpy()
        faces1 = myMesh1.faces_list()[0].cpu().numpy()
    else:
        verts1 = myMesh1.vert
        faces1 = myMesh1.face

    if hasattr(myMesh2, 'verts_list'):
        verts2 = myMesh2.verts_list()[0].cpu().numpy()
        faces2 = myMesh2.faces_list()[0].cpu().numpy()
    else:
        verts2 = myMesh2.vert
        faces2 = myMesh2.face

    # Create plots in website mode for saving
    website()
    
    # Create the plots separately using plot instead of subplot
    p = mp.plot(verts1, faces1, c=cmap1, return_plot=True)
    p1 = mp.plot(verts2, faces2, c=cmap2, return_plot=True)
    
    # Create the HTML content with flex container
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Double Plot</title>
    </head>
    <body>
        <div style="display: flex;">
            <div style="width: 50%;">{p.to_html(imports=True, html_frame=False)}</div>
            <div style="width: 50%;">{p1.to_html(imports=False, html_frame=False)}</div>
        </div>
    </body>
    </html>
    """
    
    # Ensure the directory exists
    os.makedirs('data/plots', exist_ok=True)
    
    full_save_path = os.path.join('data/plots', save_path)
    with open(full_save_path, 'w') as f:
        f.write(html_content)

    if show:
        jupyter()
        d = mp.subplot(verts1, faces1, c=cmap1, s=[2, 2, 0])
        mp.subplot(verts2, faces2, c=cmap2, s=[2, 2, 1], data=d)
    
def get_colors(vertices):
    """Get colors for vertices using their normalized positions as RGB values"""
    # If vertices is a Meshes object, get the vertices tensor and convert to numpy
    if isinstance(vertices, MeshContainer):
        vertices = vertices.vert  # Using the correct 'vert' attribute
    elif hasattr(vertices, 'verts_list'):
        vertices = vertices.verts_list()[0]
    
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()
    
    min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord)/(max_coord - min_coord)
    return cmap

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy()"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError

def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError

def convert_trimesh_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vertices, dtype=torch.float32), torch.tensor(
        tm.faces, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def convert_mesh_container_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vert, dtype=torch.float32), torch.tensor(
        tm.face, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def load_textured_mesh(mesh_path, device):
    verts, faces, aux = load_obj(mesh_path)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
    mesh = mesh.to(device)
    return mesh

def cosine_similarity(a, b):
    if len(a) > 30000:
        return cosine_similarity_batch(a, b, batch_size=30000)
    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity

def cosine_similarity_batch(a, b, batch_size=30000):
    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()
    return similarity_matrix

def hungarian_correspondence(similarity_matrix):
    # Convert similarity matrix to a cost matrix by negating the similarity values
    cost_matrix = -similarity_matrix.cpu().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a binary matrix with 1s at matched indices and 0s elsewhere
    num_rows, num_cols = similarity_matrix.shape
    match_matrix = np.zeros((num_rows, num_cols), dtype=int)
    match_matrix[row_indices, col_indices] = 1
    match_matrix = torch.from_numpy(match_matrix).cuda()
    return match_matrix

def gmm(a, b):
    # Compute Gram matrices
    gram_matrix_a = torch.mm(a, a.t())
    gram_matrix_b = torch.mm(b, b.t())

    # Expand dimensions to facilitate broadcasting
    gram_matrix_a = gram_matrix_a.unsqueeze(1)
    gram_matrix_b = gram_matrix_b.unsqueeze(0)

    # Compute Frobenius norm for each pair of vertices using vectorized operations
    correspondence_matrix = torch.norm(gram_matrix_a - gram_matrix_b, p='fro', dim=2)

    return correspondence_matrix

def load_mesh(file_path, device):
    """Load mesh, using the correct loader based on texture presence."""
    print(f"Loading mesh from {file_path}")

    # Check if the OBJ file contains texture references
    with open(file_path, "r") as f:
        contains_texture = any(line.startswith(("mtllib", "usemtl")) for line in f)

    if contains_texture:
        print("Detected texture references. Using load_objs_as_meshes.")
        return load_objs_as_meshes([file_path], device=device)
    else:
        print("No texture detected. Using MeshContainer.")
        return MeshContainer().load_from_file(file_path)

def process_3d_models(models_path, output_path, device, sam_model, dino_model, pipe, num_views, H, W, tolerance, is_folder_structure=False):
    """
    Process 3D model files and compute their features, handling both single .obj files and folder structures.
    
    Args:
        models_path (str): Path to the directory containing models
        output_path (str): Path where rendered meshes will be saved
        device: Computing device (CPU/GPU)
        sam_model: SAM model instance
        dino_model: DINO model instance
        pipe: Pipeline instance
        num_views (int): Number of views to process
        H (int): Height of the rendered image
        W (int): Width of the rendered image
        tolerance (float): Processing tolerance
        is_folder_structure (bool): Whether the models are organized in folders (default: False)
    """
    def process_single_model(file_path, object_name, prompt):
        print(f"Processing {object_name} with prompt {prompt}")
        object_mesh = load_mesh(file_path, device)
        save_path = os.path.join(output_path, f"{object_name}_rendered.pt")
        compute_features(
            device, sam_model, dino_model, pipe, 
            object_mesh, prompt, num_views, H, W, 
            tolerance, save_path
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if is_folder_structure:
        # Process models organized in folders
        for object_folder in os.listdir(models_path):
            folder_path = os.path.join(models_path, object_folder)
            if os.path.isdir(folder_path):
                object_name = object_folder.split("_")[0]
                object_file_path = os.path.join(folder_path, f"{object_folder}.obj")
                if os.path.exists(object_file_path):
                    process_single_model(object_file_path, object_folder, object_name)
    else:
        # Process individual .obj files
        for object_file in os.listdir(models_path):
            if object_file.endswith(".obj"):
                object_name = object_file.split(".")[0]
                object_file_path = os.path.join(models_path, object_file)
                object_prompt = object_name.split("_")[0]
                process_single_model(object_file_path, object_name, object_prompt)