import numpy as np
from scipy.io import loadmat
import torch
import os

from dataloaders.mesh_container import MeshContainer

def load_gt_data(gt_path, debug=False):
    """Load ground truth data from .mat file and print debug info."""
    if debug:
        print(f"\nLoading ground truth from: {gt_path}")
    mat_data = loadmat(gt_path)
    
    # Print debug information
    if debug:
        print(f"File name: {mat_data['fname'][0] if 'fname' in mat_data else 'Not found'}")
        print(f"Available points: {mat_data['points'].shape if 'points' in mat_data else 'Not found'}")
        print(f"Number of centroids: {mat_data['centroids'].shape if 'centroids' in mat_data else 'Not found'}")
        print(f"Vertex indices shape: {mat_data['verts'].shape if 'verts' in mat_data else 'Not found'}")
    
    return mat_data

def find_common_points(source_gt, target_gt, debug=False):
    """Find intersection of available points between source and target."""
    source_points = source_gt['points'].flatten()
    target_points = target_gt['points'].flatten()
    
    common_points = np.intersect1d(source_points, target_points)
    if debug:
        print(f"\nNumber of common points: {len(common_points)}")
    
    return common_points

def get_point_indices(points_array, common_points):
    """Get indices in the points array for common points."""
    indices = []
    for point in common_points:
        idx = np.where(points_array.flatten() == point)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    return indices

def compute_scale(vertices):
    """Compute the maximal Euclidean distance between any two points in the mesh."""
    max_dist = 0
    for i in range(len(vertices)):
        dists = np.linalg.norm(vertices - vertices[i], axis=1)
        max_dist = max(max_dist, np.max(dists))
    return max_dist

def compute_metrics(predicted_points, target_points, gamma=0.1):
    """
    Compute average correspondence error and accuracy.
    
    Args:
        predicted_points: numpy array of predicted point coordinates
        target_points: numpy array of ground truth point coordinates
        gamma: threshold parameter for accuracy computation (1%)
    """
    # Compute L2 distances
    distances = np.linalg.norm(predicted_points - target_points, axis=1)
    
    # Compute average error
    avg_error = np.mean(distances)
    
    # Compute scale (d) as the maximal Euclidean distance between any two points in target
    scale = compute_scale(target_points)
    
    # Compute accuracy
    threshold = gamma * scale
    accuracy = np.mean(distances < threshold)
    
    return avg_error, accuracy, distances, scale

def evaluate_correspondence(source_gt, target_gt, predicted_mapping, source_vertices, target_vertices, debug=False):
    """Evaluate correspondence accuracy and error."""
    # Find common points
    common_points = find_common_points(source_gt, target_gt, debug)
    if debug:
        print(f"\nSource gt points: {source_gt['points'].tolist()}")
        print(f"Target gt points: {target_gt['points'].tolist()}")
        print(f"Common points: {[point for point in common_points]}")
    
    # Get indices for common points in both meshes
    source_indices = get_point_indices(source_gt['points'], common_points)
    target_indices = get_point_indices(target_gt['points'], common_points)
    if debug:
        print(f"\nSource indices: {source_indices}")
        print(f"Target indices: {target_indices}")
    
    source_gt_verts = source_gt['verts'][source_indices].flatten()
    target_gt_verts = target_gt['verts'][target_indices].flatten()
    if debug:
        print(f"\nSource gt vertices: {source_gt_verts.shape}")
        print(f"Source gt vertices: {source_gt_verts}")
        print(f"Target gt vertices: {target_gt_verts.shape}")
        print(f"Target gt vertices: {target_gt_verts}")
    
    # Get predicted vertices for source points
    predicted_verts = predicted_mapping[source_gt_verts]
    if debug:
        print(f"\nPredicted vertices: {predicted_verts.shape}")

    # Get actual 3D coordinates
    predicted_coords = target_vertices[predicted_verts]
    target_coords = target_vertices[target_gt_verts]
    if debug:
        print(f"\nPredicted coords: {predicted_coords.shape}")
        print(f"Target coords: {target_coords.shape}")
    
    # Compute metrics
    avg_error, accuracy, distances, scale = compute_metrics(predicted_coords, target_coords)
    
    # These prints are always shown (not debug)
    print("\nEvaluation Results:")
    print(f"Number of points evaluated: {len(common_points)}")
    print(f"Scale (d): {scale:.6f}")
    print(f"Average correspondence error (err): {avg_error:.6f}")
    print(f"Correspondence accuracy (acc, γ=1%): {accuracy:.6f}")
    
    return avg_error, accuracy, distances

def main():
    debug = True  # Add debug flag
    
    # Paths
    source_file_path = "shrec20_dataset/SHREC20b_lores/models/cow.obj"
    target_file_path = "shrec20_dataset/SHREC20b_lores/models/camel_a.obj"
    cow_gt_path = 'shrec20_dataset/SHREC20b_lores_gts/cow.mat'
    camel_gt_path = 'shrec20_dataset/SHREC20b_lores_gts/camel_a.mat'
    mapping_path = 'predicted_mapping.npy'  # Path to saved mapping from test.ipynb
    
    # Load ground truth data
    source_gt = load_gt_data(cow_gt_path, debug)
    target_gt = load_gt_data(camel_gt_path, debug)
    
    # Load predicted mapping
    predicted_mapping = np.load(mapping_path)
    if debug:
        print(f"\nLoaded predicted mapping shape: {predicted_mapping.shape}")
    
    # Load source and target mesh vertices
    source_mesh = MeshContainer().load_from_file(source_file_path)
    target_mesh = MeshContainer().load_from_file(target_file_path)
    if debug:
        print("\nMesh info:")
        print(f"Source - Vertices: {len(source_mesh.vert)}, Faces: {len(source_mesh.face)}")
        print(f"Target - Vertices: {len(target_mesh.vert)}, Faces: {len(target_mesh.face)}")
    source_vertices = np.array(source_mesh.vert)
    target_vertices = np.array(target_mesh.vert)
    if debug:
        print(f"Source vertices shape: {source_vertices.shape}")
        print(f"Target vertices shape: {target_vertices.shape}")
    
    # Evaluate correspondence
    evaluate_correspondence(source_gt, target_gt, predicted_mapping, source_vertices, target_vertices, debug)

if __name__ == "__main__":
    main()