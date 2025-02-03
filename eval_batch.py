import torch
import numpy as np
from pathlib import Path
from dataloaders.mesh_container import MeshContainer
from eval import evaluate_meshes
from pytorch3d.io import load_objs_as_meshes
import os
from utils import compute_features, cosine_similarity, load_mesh


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
        # Construct file paths
        models_path = Path(base_path) / "models"
        gts_path = Path(base_path + "_gts")

        source_file_path = str(models_path / f"{source_name}.obj")
        target_file_path = str(models_path / f"{target_name}.obj")
        source_gt_path = str(gts_path / f"{source_name}.mat")
        target_gt_path = str(gts_path / f"{target_name}.mat")

        # Load meshes
        source_mesh = load_mesh(source_file_path, device)
        target_mesh = load_mesh(target_file_path, device)

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
        s = cosine_similarity(f_target.to(device),f_source.to(device))        
        s = torch.argmax(s, dim=0).cpu().numpy()
        mapping_path = f"data/mappings/{source_name}_{target_name}_mapping.npy"
        Path("data/mappings").mkdir(exist_ok=True)
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
            avg_error_display = f"{result['avg_error']:.2f}".replace(".", ",")  
            accuracy_display = f"{result['accuracy'] * 100:.2f}".replace(".", ",")
            print(f"Average correspondence error (err): {avg_error_display}")
            print(f"Correspondence accuracy (acc, γ=1%): {accuracy_display}%")
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
        print(f"Average error across all pairs: {successful_results['avg_error'].mean()* 100:.2f}")
        print(f"Average accuracy across all pairs: {successful_results['accuracy'].mean():.2f}")
        print(f"Successfully processed {len(successful_results)} out of {len(results)} pairs")

if __name__ == "__main__":
    import torch
    import time


    start_time = time.time()  # Start timing

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("No GPU/MPS available, falling back to CPU.")

    from diffusion import init_pipe
    from utils import cosine_similarity, double_plot, get_colors
    from dino import init_dino
    from sam2_setup import init_sam2
    from utils import compute_features, load_mesh
    import numpy as np
    num_views = 1
    H = 512
    W = 512
    tolerance = 0.004
    use_normal_map = True
    num_images_per_prompt = 1
    bq = True
    use_sam = False
    use_only_diffusion = False
    use_diffusion = True
    is_tosca = False

    save_path=None # if not None, save batched_renderings, normal_batched_renderings, camera, depth to 'rendered_mesh_output.pt'
   
    sam_model = None
    pipe = init_pipe(device)
    dino_model = init_dino(device)


    results = run_batch_evaluation(
        pairs_file='data/SHREC20b_lores/test-sets/test-set2.txt',
        base_path="data/SHREC20b_lores",
        device=device,
        sam_model=sam_model,
        dino_model=dino_model,
        pipe=pipe,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance
    )


    end_time = time.time()
    elapsed_time = end_time - start_time 

    # Format the output
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")