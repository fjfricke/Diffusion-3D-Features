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

def compute_metrics(predicted_points, target_points, scale, gamma=0.01):
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
    
    # source_gt_coords = source_gt['centroids'][source_indices]
    # target_gt_coords = target_gt['centroids'][target_indices]
    # if debug:
    #     print(f"\nSource gt coords: {source_gt_coords}")
    #     print(f"Target gt coords: {target_gt_coords}")

    source_centroids = source_gt['centroids']
    source_coords = source_vertices[source_gt['verts'] - 1].reshape(-1, 3)
    # print(source_centroids - source_coords)
    print("Sanity check:")
    print((source_centroids - source_coords).max())

    source_gt_verts = source_gt['verts'][source_indices].flatten() - 1
    target_gt_verts = target_gt['verts'][target_indices].flatten() - 1
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
    target_coords = target_gt['centroids'][target_indices]
    if debug:
        print(f"\nPredicted coords: {predicted_coords.shape}")
        print(f"Target coords: {target_coords.shape}")
    
    # Compute metrics
    scale = compute_scale(target_vertices)
    avg_error, accuracy, distances, scale = compute_metrics(predicted_coords, target_coords, scale, gamma=0.01)
    
    avg_error_display = f"{avg_error:.2f}".replace(".", ",")  
    accuracy_display = f"{accuracy * 100:.2f}".replace(".", ",")

    if debug:
        print("\nEvaluation Results:")
        print(f"Number of points evaluated: {len(common_points)}")
        print(f"Scale (d): {scale:.6f}")
        print(f"Average correspondence error (err): {avg_error_display}")
        print(f"Correspondence accuracy (acc, γ=1%): {accuracy_display}%")
    
    return avg_error, accuracy, distances

def evaluate_meshes(source_file_path, target_file_path, source_gt_path, target_gt_path, mapping_path, debug=False):
    """
    Evaluate correspondence between two meshes using ground truth data.
    
    Args:
        source_file_path (str): Path to source mesh obj file
        target_file_path (str): Path to target mesh obj file
        source_gt_path (str): Path to source ground truth mat file
        target_gt_path (str): Path to target ground truth mat file
        mapping_path (str): Path to predicted mapping npy file
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (average_error, accuracy, distances)
    """
    # Load ground truth data
    source_gt = load_gt_data(source_gt_path, debug)
    target_gt = load_gt_data(target_gt_path, debug)
    
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
    return evaluate_correspondence(source_gt, target_gt, predicted_mapping, source_vertices, target_vertices, debug)

def main():
    debug = True
    
    # Paths
    source_file_path = "data/SHREC20b_lores/models/cow.obj"
    target_file_path = "data/SHREC20b_lores/models/hippo.obj"
    cow_gt_path = 'data/SHREC20b_lores_gts/hippo.mat'
    camel_gt_path = 'data/SHREC20b_lores_gts/hippo.mat'
    mapping_path = 'predicted_mapping.npy'  # Path to saved mapping from test.ipynb
    
    # Call the evaluation function
    evaluate_meshes(source_file_path, target_file_path, cow_gt_path, camel_gt_path, mapping_path, debug)

if __name__ == "__main__":
    main()