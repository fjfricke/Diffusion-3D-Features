import torch
import numpy as np
from pathlib import Path
from eval import evaluate_meshes
from utils import compute_features_dinotracker, cosine_similarity, load_mesh


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
        num_views,
        H,
        W,
        tolerance,
        use_normal_map=True,
        bq=True,
        is_tex=False,
        dinotracker_path=None,
        use_only_dino=False
    ):
    """Process a single pair of meshes."""
    try:
        # Construct file paths
        models_path = Path(base_path) / "models"
        gts_path = Path(base_path + "_gts")
        tex_path = Path(base_path + "_tex")  # New path for textured models

        source_file_path = str(models_path / f"{source_name}.obj")
        target_file_path = str(models_path / f"{target_name}.obj")
        source_gt_path = str(gts_path / f"{source_name}.mat")
        target_gt_path = str(gts_path / f"{target_name}.mat")

        # Add paths for textured meshes
        source_tex_file_path = str(tex_path / f"{source_name}_tex" / f"{source_name}_tex.obj")
        target_tex_file_path = str(tex_path / f"{target_name}_tex" / f"{target_name}_tex.obj")

        # Load meshes
        source_mesh = load_mesh(source_file_path, device)
        target_mesh = load_mesh(target_file_path, device)

        if use_only_dino:
            embedding_filename = "dino_embed_video.pt"
        else:
            embedding_filename = "refined_embeddings.pt"

        # get dinotracker path for features
        if not is_tex:
            dinotracker_path_source = Path(dinotracker_path) / f"{source_name}_rendered/dino_embeddings/{embedding_filename}"
            dinotracker_path_target = Path(dinotracker_path) / f"{target_name}_rendered/dino_embeddings/{embedding_filename}"
        else:
            dinotracker_path_source = Path(dinotracker_path) / f"{source_name}_tex_rendered/dino_embeddings/{embedding_filename}"
            dinotracker_path_target = Path(dinotracker_path) / f"{target_name}_tex_rendered/dino_embeddings/{embedding_filename}"


        f_source = compute_features_dinotracker(device, dinotracker_path_source, source_mesh, source_name, num_views, H, W, tolerance, use_normal_map, bq, is_tex)
        
        f_target = compute_features_dinotracker(device, dinotracker_path_target, target_mesh, target_name, num_views, H, W, tolerance, use_normal_map, bq, is_tex)

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
    base_path,
    device,
    num_views,
    H,
    W,
    tolerance,
    use_normal_map,
    bq,
    is_tex,
    dinotracker_path,
    use_only_dino=False
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
            num_views,
            H,
            W,
            tolerance,
            use_normal_map,
            bq,
            is_tex,
            dinotracker_path,
            use_only_dino
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

    from utils import cosine_similarity
    from utils import load_mesh
    import numpy as np


    results = run_batch_evaluation(
        pairs_file='data/SHREC20b_lores/test-sets/test-set5.txt',
        base_path="data/SHREC20b_lores",
        device=device,
        num_views=50,
        H=512,
        W=512,
        tolerance=0.004,
        use_normal_map=True,
        bq=True,
        is_tex=True,
        dinotracker_path="/workspace/dino-tracker/dataset",
        use_only_dino=True
    )


    end_time = time.time()
    elapsed_time = end_time - start_time 

    # Format the output
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")