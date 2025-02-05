# Project Overview


This project is based on the original repository: [Diffusion-3D-Features](https://github.com/niladridutt/Diffusion-3D-Features).


Installation
Please refer to our `environment.yml` file for installation instructions.
Note: For `xformers`, ensure to install it separately after the initial setup.

## Data

The project data is available on Google Drive: [Project Data](https://drive.google.com/drive/folders/1C6lFfCbwQqxlvUE8niVfbeIzeblCnXcx?usp=share_link)

The dataset includes:
- Original 3D meshes from SHREC20b
- Textured versions of the meshes (generated using [Text2Tex](https://github.com/daveredrum/Text2Tex))
- Ground truth correspondence data
- Data used for DINO tracker (both textured and untextured)
- Results mentioned in the paper


## Sample Usage


A complete example workflow is provided in `test_correspondence.ipynb`. This notebook demonstrates:
- Setting up the environment and loading models
- Loading and processing 3D meshes (both textured and non-textured)
- Extracting features using DINO and other models
- Computing correspondences between meshes
- Evaluating the accuracy of the correspondences
- Visualizing the results

The notebook serves as a practical guide for using the pipeline with sample data.

## Code Structure

### Key Files and Their Functions

1. **`diff3f.py`**: 
   - Contains functions for feature extraction from 3D meshes using diffusion, SAM, and DINO.
   - **DINO Tracker Data**: To obtain data for the DINO tracker pipeline, use the following functionality:
     ```python
     if save_path:
         os.makedirs(os.path.dirname(save_path), exist_ok=True)
         torch.save({
             'renderings': batched_renderings,
             'normal_renderings': normal_batched_renderings,
             'camera': camera,
             'depth': depth
         }, save_path)
         print(f"Rendered mesh saved to {save_path}")
     ```

2. **`eval.py`**:
   - Provides functionality to evaluate the correspondences between two meshes.
   - We use the average correspondence error and the correspondence accuracy as our evaluation criteria.
   - Key functions include `evaluate_correspondence` and `evaluate_meshes`.

3. **`eval_batch.py`**:
   - Facilitates batch processing of mesh pairs for evaluation.
   - Key function: `run_batch_evaluation` which processes multiple mesh pairs and evaluates them.

4. **`utils.py`**:
   - Contains utility functions for mesh processing, feature computation, and visualization.
   - Key functions include `compute_features`, `cosine_similarity`, and `load_mesh`.
   - The script uses pairs from the data/SHREC20b_lores/test-sets folder and saves the results in a CSV file.
   - Test set 5 combines the other 4 test sets and is used in our evaluation.

5. **`mesh_video_generator.py`**:
   - Provides functionality to generate videos of rotating 3D meshes.
   - Features include:
     - Rendering meshes from multiple viewpoints (azimuth and elevation rotations)
     - Support for normal maps and depth information
     - Batch processing of multiple meshes
     - Video generation 
   - Example usage:
     ```python
     generator = MeshVideoGenerator(
         output_dir="outputs",
         hw=128,  # height/width of output
         num_views=10,  # number of viewpoints
         use_normal_map=True
     )
     generator.process_folder("path/to/meshes")
     ```