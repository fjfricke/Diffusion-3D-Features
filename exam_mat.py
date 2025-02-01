import scipy

def load_mat_file(file_path):
    """Load and examine a .mat file containing ground-truth data."""
    try:
        data = scipy.io.loadmat(file_path)
        
        # Extract key variables
        fname = data.get('fname', None)
        points = data.get('points', None)
        centroids = data.get('centroids', None)
        baryc = data.get('baryc', None)
        verts = data.get('verts', None)
        
        # Print information about each variable
        print("File Name:", fname)
        print("\nPoints (Indices of Locations on the Mesh):")
        print(points)
        print("\nCentroids (Medoid Locations):")
        print(centroids)
        print("\nBarycentric Coordinates:")
        print(baryc)
        print("\nClosest Vertex Indices:")
        print(verts)
        
        return data
    except Exception as e:
        print("Error loading .mat file:", e)
        return None

# Example usage (Replace 'your_file.mat' with actual file path)
cow_gt_path = 'shrec20_dataset/SHREC20b_lores_gts/cow.mat'
file_path = cow_gt_path
data = load_mat_file(file_path)