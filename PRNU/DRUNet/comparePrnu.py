import numpy as np
from scipy.stats import pearsonr
import os

# === INPUT: absolute paths to the two PRNU fingerprint files ===
path_cam1 = r'X:\Projects\Open set source camera identification\Fingerprints\Agfa_DC-504_0\Agfa_DC-504_0_fingerprint.npy'
path_cam2 = r'X:\Projects\Open set source camera identification\Fingerprints\Agfa_DC-733s\Agfa_DC-733s_fingerprint.npy'

# === Load the PRNU fingerprint arrays ===
def load_fingerprint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)

# === Resize the smaller fingerprint to match the larger one (if needed) ===
def resize_to_match(f1, f2):
    if f1.shape != f2.shape:
        min_shape = tuple(map(min, zip(f1.shape, f2.shape)))
        f1 = f1[:min_shape[0], :min_shape[1]]
        f2 = f2[:min_shape[0], :min_shape[1]]
    return f1, f2

# === Compute normalized cross-correlation (Pearson) ===
def compute_similarity(f1, f2):
    flat1 = f1.flatten()
    flat2 = f2.flatten()
    corr, _ = pearsonr(flat1, flat2)
    return corr

# === Main logic ===
def compare_prnu(path1, path2):
    f1 = load_fingerprint(path1)
    f2 = load_fingerprint(path2)
    
    f1, f2 = resize_to_match(f1, f2)

    similarity_score = compute_similarity(f1, f2)

    # Scale similarity score from [-1, 1] to [0, 100]%
    similarity_percent = max(0.0, min((similarity_score + 1) * 50, 100))
    dissimilarity_percent = 100 - similarity_percent

    # Decision rule (you can tune threshold)
    threshold = 60.0  # you can adjust this based on experiments
    same_camera = similarity_percent >= threshold

    print(f"Similarity: {similarity_percent:.2f}%")
    print(f"Dissimilarity: {dissimilarity_percent:.2f}%")
    print(f"Result: {'SAME camera' if same_camera else 'DIFFERENT cameras'}")

# === Run the comparison ===
compare_prnu(path_cam1, path_cam2)
