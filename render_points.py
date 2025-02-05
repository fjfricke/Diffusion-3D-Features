import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
import os

from dataloaders.mesh_container import MeshContainer

def load_gt_data(gt_path, debug=False):
    """Load ground truth data from a .mat file."""
    if debug:
        print(f"\nLoading ground truth from: {gt_path}")
    mat_data = loadmat(gt_path)
    if debug:
        print(f"File name: {mat_data.get('fname', 'Not found')}")
        print(f"Points shape: {mat_data.get('points', 'Not found')}")
        print(f"Centroids shape: {mat_data.get('centroids', 'Not found')}")
        print(f"Verts shape: {mat_data.get('verts', 'Not found')}")
    return mat_data

def find_common_points(source_gt, target_gt, debug=False):
    """
    Find the intersection of available points between source and target.
    Assumes the ground truth .mat files contain a key 'points' with point indices.
    """
    source_points = source_gt['points'].flatten()
    target_points = target_gt['points'].flatten()
    common_points = np.intersect1d(source_points, target_points)
    if debug:
        print(f"Number of common points: {len(common_points)}")
    return common_points

def get_point_indices(points_array, common_points):
    """Return the indices in the given points_array corresponding to the common_points."""
    indices = []
    for point in common_points:
        idx = np.where(points_array.flatten() == point)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    return np.array(indices)

def render_interactive(source_vertices, target_vertices, source_common, 
                       gt_target_coords, predicted_target_coords):
    """
    Render the source and target meshes with the evaluated common points using Plotly.
    
    Parameters:
      source_vertices: numpy array (N_source x 3) of source mesh vertices.
      target_vertices: numpy array (N_target x 3) of target mesh vertices.
      source_common: numpy array (n_common x 3) of common points on the source mesh.
      gt_target_coords: numpy array (n_common x 3) of ground truth target coordinates.
      predicted_target_coords: numpy array (n_common x 3) of predicted target coordinates.
    """
    # Create a color array for the points with better separation
    n_points = len(source_common)
    
    # Generate distinct colors using golden ratio method
    golden_ratio = (1 + 5 ** 0.5) / 2
    colors = [
        f'hsl({((i * golden_ratio * 360) % 360):.1f}, 70%, {50 + (i % 2) * 10}%)'
        for i in range(n_points)
    ]
    
    # Shuffle the colors to ensure neighboring points have different colors
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(colors)

    # Create hover text arrays
    source_hover_text = [f"Point {i}" for i in range(n_points)]
    gt_hover_text = [f"GT Point {i}" for i in range(n_points)]
    pred_hover_text = [f"Predicted Point {i}" for i in range(n_points)]
    
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Source Mesh with Common Points", 
                                        "Target Mesh with GT & Predicted Points"))

    # Source Mesh trace (all vertices in light gray)
    fig.add_trace(
        go.Scatter3d(
            x=source_vertices[:, 0],
            y=source_vertices[:, 1],
            z=source_vertices[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray'),
            name='Source Mesh',
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Common points on the source mesh (colored)
    fig.add_trace(
        go.Scatter3d(
            x=source_common[:, 0],
            y=source_common[:, 1],
            z=source_common[:, 2],
            mode='markers',
            marker=dict(
                size=5, 
                color=colors,
                line=dict(
                    width=2,
                    color='white'
                )
            ),
            name='Common Points',
            text=source_hover_text,
            customdata=np.arange(n_points),  # Add index data
            hovertemplate="<b>%{text}</b><br>" +
                         "x: %{x:.3f}<br>" +
                         "y: %{y:.3f}<br>" +
                         "z: %{z:.3f}<br>" +
                         "<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Target Mesh trace (all vertices in light gray)
    fig.add_trace(
        go.Scatter3d(
            x=target_vertices[:, 0],
            y=target_vertices[:, 1],
            z=target_vertices[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray'),
            name='Target Mesh',
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Ground truth target points (colored circles)
    gt_trace = go.Scatter3d(
        x=gt_target_coords[:, 0],
        y=gt_target_coords[:, 1],
        z=gt_target_coords[:, 2],
        mode='markers',
        marker=dict(
            size=5,  # Use single value instead of array
            color=colors, 
            symbol='circle',
            line=dict(
                width=2,  # Use single value instead of array
                color='white'
            )
        ),
        name='GT Target Points',
        text=gt_hover_text,
        customdata=np.arange(n_points),  # Add index data
        hovertemplate="<b>%{text}</b><br>" +
                     "x: %{x:.3f}<br>" +
                     "y: %{y:.3f}<br>" +
                     "z: %{z:.3f}<br>" +
                     "<extra></extra>"
    )
    
    # Predicted target points (colored diamonds)
    pred_trace = go.Scatter3d(
        x=predicted_target_coords[:, 0],
        y=predicted_target_coords[:, 1],
        z=predicted_target_coords[:, 2],
        mode='markers',
        marker=dict(
            size=5,  # Use single value instead of array
            color=colors, 
            symbol='diamond',
            line=dict(
                width=2,  # Use single value instead of array
                color='white'
            )
        ),
        name='Predicted Points',
        text=pred_hover_text,
        customdata=np.arange(n_points),  # Add index data
        hovertemplate="<b>%{text}</b><br>" +
                     "x: %{x:.3f}<br>" +
                     "y: %{y:.3f}<br>" +
                     "z: %{z:.3f}<br>" +
                     "<extra></extra>"
    )

    fig.add_trace(gt_trace, row=1, col=2)
    fig.add_trace(pred_trace, row=1, col=2)

    # Update layout for a better interactive experience
    fig.update_layout(
        height=700, 
        width=1200,
        title_text="Interactive 3D Visualization",
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        scene2=dict(  # Add specific settings for the target mesh scene
            aspectmode="data",
            hovermode='closest',
            dragmode='turntable',
        ),
        scene=dict(  # Add settings for the source mesh scene
            aspectmode="data",
            hovermode='closest',
            dragmode='turntable',
        )
    )

    # Add JavaScript for point highlighting
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [{'scene.camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}}],
                    'label': 'Reset View',
                    'method': 'relayout'
                }
            ],
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'y': 1.1,
        }]
    )

    # Add highlighting instructions
    fig.add_annotation(
        text="Hover over points to see point indices",
        xref="paper", yref="paper",
        x=0, y=1.05,
        showarrow=False,
        font=dict(size=14)
    )

    # Show the interactive figure
    fig.show(config={
        'editable': False,
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'showLink': False,
        'responsive': True
    })

def main(debug=True):
    # Datapaths (using the ones you provided)
    source_file_path = "data/SHREC20b_lores/models/hippo.obj"
    target_file_path = "data/SHREC20b_lores/models/cow.obj"
    cow_gt_path = 'data/SHREC20b_lores_gts/hippo.mat'
    camel_gt_path = 'data/SHREC20b_lores_gts/cow.mat'
    mapping_path = 'predicted_mapping.npy'  # predicted mapping (np.load)

    # --- Load ground truth data ---
    source_gt = load_gt_data(cow_gt_path, debug=debug)
    target_gt = load_gt_data(camel_gt_path, debug=debug)
    
    # --- Load predicted mapping ---
    predicted_mapping = np.load(mapping_path)
    if debug:
        print(f"Loaded predicted mapping with shape: {predicted_mapping.shape}")
    
    # --- Load meshes ---
    source_mesh = MeshContainer().load_from_file(source_file_path)
    target_mesh = MeshContainer().load_from_file(target_file_path)
    source_vertices = np.array(source_mesh.vert)
    target_vertices = np.array(target_mesh.vert)
    if debug:
        print(f"Source vertices shape: {source_vertices.shape}")
        print(f"Target vertices shape: {target_vertices.shape}")
    
    # --- Find common points ---
    common_points = find_common_points(source_gt, target_gt, debug=debug)
    
    # Get indices for these common points in the ground truth 'points'
    source_point_indices = get_point_indices(source_gt['points'], common_points)
    target_point_indices = get_point_indices(target_gt['points'], common_points)
    
    if debug:
        print(f"Source common point indices: {source_point_indices}")
        print(f"Target common point indices: {target_point_indices}")
    
    # --- Retrieve source common points (from mesh vertices) ---
    # Here we assume that the ground truth for source uses vertex indices in source_gt['verts'] (MATLAB indexing).
    source_gt_verts = source_gt['verts'][source_point_indices].flatten() - 1  # convert to 0-indexing
    source_common = source_vertices[source_gt_verts]
    
    # --- Retrieve target ground truth coordinates ---
    # In your case, target GT coordinates come from the centroids.
    gt_target_coords = target_gt['centroids'][target_point_indices]
    
    # --- Compute predicted target coordinates ---
    # Use source GT vertices (converted to 0-indexing) to index the predicted mapping.
    source_gt_verts_for_mapping = source_gt['verts'][source_point_indices].flatten() - 1
    predicted_target_verts = predicted_mapping[source_gt_verts_for_mapping]
    predicted_target_coords = target_vertices[predicted_target_verts]
    
    # --- Render interactively using Plotly ---
    render_interactive(source_vertices, target_vertices, source_common,
                       gt_target_coords, predicted_target_coords)

if __name__ == "__main__":
    main(debug=True)
