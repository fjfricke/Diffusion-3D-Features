import numpy as np
import torch
import scipy.io as sio


def load_shrec_ground_truth(mat_file_path):
    """Load ground truth correspondences from SHREC'20 .mat file"""
    mat_data = sio.loadmat(mat_file_path)
    return {
        "fname": mat_data["fname"],
        "points": mat_data["points"].squeeze(),
        "centroids": mat_data["centroids"],
        "verts": mat_data["verts"].squeeze(),
        "baryc": mat_data["baryc"],
    }


def evaluate_shrec_correspondence(
    source_vertices,
    target_vertices,
    correspondences,
    source_gt,
    target_gt,
    threshold=0.01,
):
    """
    Evaluate shape correspondence using SHREC'20 ground truth format.
    Assumes correspondences map from target to source vertices.

    Parameters:
    -----------
    source_vertices: np.ndarray
        Vertices of source mesh (N x 3)
    target_vertices: np.ndarray
        Vertices of target mesh (M x 3)
    correspondences: np.ndarray
        Correspondence indices mapping target vertices to source vertices (M,)
    source_gt: dict
        Ground truth data for source mesh
    target_gt: dict
        Ground truth data for target mesh
    threshold: float
        Distance threshold for considering a correspondence accurate
    """
    # Convert to numpy if needed
    if torch.is_tensor(source_vertices):
        source_vertices = source_vertices.cpu().numpy()
    if torch.is_tensor(target_vertices):
        target_vertices = target_vertices.cpu().numpy()
    if torch.is_tensor(correspondences):
        correspondences = correspondences.cpu().numpy()

    # Validate input shapes
    assert source_vertices.shape[1] == 3, "Source vertices should be Nx3"
    assert target_vertices.shape[1] == 3, "Target vertices should be Mx3"
    assert len(correspondences) == len(
        target_vertices
    ), "Correspondence length should match target vertices"
    assert np.max(correspondences) < len(
        source_vertices
    ), "Correspondence indices exceed source vertex count"

    # Filter out topologically inconsistent points
    inconsistent_points = {3, 52, 53, 54, 4, 5, 6, 55, 56}
    valid_points = np.array(
        [
            i
            for i, point in enumerate(source_gt["points"])
            if point not in inconsistent_points
        ]
    )

    # Get ground truth vertex indices (0-based indexing)
    source_gt_verts = source_gt["verts"][valid_points] - 1
    target_gt_verts = target_gt["verts"][valid_points] - 1

    # For each ground truth source vertex, find the target vertex that maps to it
    source_to_target = np.where(
        correspondences[:, None] == source_gt_verts[None, :],
        np.arange(len(correspondences))[:, None],
        -1,
    ).max(axis=0)

    # Compute distances between predicted and ground truth target vertices
    predicted_points = target_vertices[source_to_target]
    gt_points = target_vertices[target_gt_verts]

    # Get mesh diagonal for normalization
    bbox = np.ptp(target_vertices, axis=0)
    mesh_diagonal = np.sqrt(np.sum(bbox**2))

    # Compute normalized distances
    distances = np.sqrt(np.sum((predicted_points - gt_points) ** 2, axis=1))
    distances_normalized = distances / mesh_diagonal

    metrics = {
        "average_error": float(np.mean(distances)),
        "accuracy": float(np.mean(distances_normalized < threshold)),
        "max_error": float(np.max(distances_normalized)),
        "min_error": float(np.min(distances_normalized)),
        "median_error": float(np.median(distances_normalized)),
        "std_error": float(np.std(distances_normalized)),
        "num_valid_points": len(valid_points),
        "mesh_diagonal": float(mesh_diagonal),
    }

    return metrics


def evaluate_cow_camel(
    cow_mesh_verts, camel_mesh_verts, correspondences, cow_gt_path, camel_gt_path
):
    """
    Evaluate correspondence between cow and camel meshes.
    Assumes correspondences map from camel (target) to cow (source).
    """
    print(
        f"Mesh sizes: Source (cow): {cow_mesh_verts.shape}, Target (camel): {camel_mesh_verts.shape}"
    )
    print(f"Correspondence array shape: {correspondences.shape}")
    print(f"Sample correspondences: {correspondences[:5]}")

    cow_gt = load_shrec_ground_truth(cow_gt_path)
    camel_gt = load_shrec_ground_truth(camel_gt_path)

    return evaluate_shrec_correspondence(
        source_vertices=cow_mesh_verts,
        target_vertices=camel_mesh_verts,
        correspondences=correspondences,
        source_gt=cow_gt,
        target_gt=camel_gt,
    )
