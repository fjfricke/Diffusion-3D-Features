import torch
import numpy as np
from pathlib import Path
from dataloaders.mesh_container import MeshContainer
from eval import evaluate_meshes
from pytorch3d.io import load_objs_as_meshes

from utils import compute_features, cosine_similarity



def read_pairs(pairs_file):
    """Read pairs from text file."""
    with open(pairs_file, "r") as f:
        pairs = [line.strip().split(",") for line in f if line.strip()]
    return pairs


def process_pair(
    source_name,
    target_name,
    base_path,
    device,
    sam_model,
    dino_model,
    pipe,
    num_views,
    H,
    W,
    tolerance,
):
    """Process a single pair of meshes."""
    # Construct file paths
    models_path = Path(base_path) / "models"
    gts_path = Path(base_path + "_gts")

    source_file_path = str(models_path / f"{source_name}.obj")
    target_file_path = str(models_path / f"{target_name}.obj")
    source_gt_path = str(gts_path / f"{source_name}.mat")
    target_gt_path = str(gts_path / f"{target_name}.mat")

    try:
        # Load meshes
        try:
            source_mesh = MeshContainer().load_from_file(source_file_path)
            target_mesh = MeshContainer().load_from_file(target_file_path)
        except NameError:
            source_mesh = load_objs_as_meshes([source_file_path], device=device)
            target_mesh = load_objs_as_meshes([target_file_path], device=device)

        # Compute features
        f_source = compute_features(
            device,
            sam_model,
            dino_model,
            pipe,
            source_mesh,
            source_name,
            num_views,
            H,
            W,
            tolerance,
        )
        f_target = compute_features(
            device,
            sam_model,
            dino_model,
            pipe,
            target_mesh,
            target_name,
            num_views,
            H,
            W,
            tolerance,
        )

        # Compute similarity and save mapping
        s = cosine_similarity(f_source.to(device), f_target.to(device))
        s = torch.argmax(s, dim=0).cpu().numpy()
        mapping_path = f"mappings/{source_name}_{target_name}_mapping.npy"
        Path("mappings").mkdir(exist_ok=True)
        np.save(mapping_path, s)

        # Evaluate
        avg_error, accuracy, distances = evaluate_meshes(
            source_file_path=source_file_path,
            target_file_path=target_file_path,
            source_gt_path=source_gt_path,
            target_gt_path=target_gt_path,
            mapping_path=mapping_path,
            debug=False,
        )

        return {
            "pair": f"{source_name}->{target_name}",
            "avg_error": avg_error,
            "accuracy": accuracy,
            "status": "success",
        }

    except Exception as e:
        return {
            "pair": f"{source_name}->{target_name}",
            "error": str(e),
            "status": "failed",
        }


def run_batch_evaluation(
    pairs_file,
    base_path="data/SHREC20b_lores",
    device="cuda",
    sam_model=None,
    dino_model=None,
    pipe=None,
    num_views=8,
    H=224,
    W=224,
    tolerance=0.2,
):
    """Run evaluation on all pairs in the dataset."""
    # Read pairs
    pairs = read_pairs(pairs_file)

    # Process each pair and collect results
    results = []
    for source_name, target_name in pairs:
        print(f"\nProcessing pair: {source_name} -> {target_name}")

        result = process_pair(
            source_name,
            target_name,
            base_path,
            device,
            sam_model,
            dino_model,
            pipe,
            num_views,
            H,
            W,
            tolerance,
        )
        results.append(result)

        # Print results for this pair
        if result["status"] == "success":
            print(f"Average correspondence error (err): {result['avg_error']:.6f}")
            print(f"Correspondence accuracy (acc, γ=1%): {result['accuracy']:.6f}")
        else:
            print(f"Failed to process pair: {result['error']}")

    # Save summary results
    save_results(results)

    return results


def save_results(results):
    """Save results to a CSV file."""
    import pandas as pd
    from datetime import datetime

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results_{timestamp}.csv", index=False)

    # Print summary statistics for successful evaluations
    successful_results = df[df["status"] == "success"]
    if not successful_results.empty:
        print("\nSummary Statistics:")
        print(
            f"Average error across all pairs: {successful_results['avg_error'].mean():.6f}"
        )
        print(
            f"Average accuracy across all pairs: {successful_results['accuracy'].mean():.6f}"
        )
        print(
            f"Successfully processed {len(successful_results)} out of {len(results)} pairs"
        )
