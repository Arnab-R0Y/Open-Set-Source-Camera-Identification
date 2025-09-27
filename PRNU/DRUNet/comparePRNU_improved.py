import numpy as np
from scipy.stats import pearsonr
import os
import glob

# === INPUT: paths to fingerprint directories ===
fingerprints_dir_old = r'X:\Projects\Open set source camera identification\Fingerprints'
fingerprints_dir_new = r'X:\Projects\Open set source camera identification\Fingerprints_Laptop'

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

# === Compare fingerprints between different cameras ===
def compare_different_cameras(fingerprints_dir):
    print(f"\n=== Comparing fingerprints from: {fingerprints_dir} ===")
    
    # Get all fingerprint files
    fingerprint_files = glob.glob(os.path.join(fingerprints_dir, "*", "*_fingerprint.npy"))
    
    if len(fingerprint_files) < 2:
        print("Need at least 2 fingerprint files to compare")
        return
    
    similarities = []
    camera_pairs = []
    
    # Compare each pair of different cameras
    for i in range(len(fingerprint_files)):
        for j in range(i + 1, len(fingerprint_files)):
            cam1_path = fingerprint_files[i]
            cam2_path = fingerprint_files[j]
            
            cam1_name = os.path.basename(os.path.dirname(cam1_path))
            cam2_name = os.path.basename(os.path.dirname(cam2_path))
            
            try:
                f1 = load_fingerprint(cam1_path)
                f2 = load_fingerprint(cam2_path)
                
                f1, f2 = resize_to_match(f1, f2)
                similarity_score = compute_similarity(f1, f2)
                
                # Scale similarity score from [-1, 1] to [0, 100]%
                similarity_percent = max(0.0, min((similarity_score + 1) * 50, 100))
                
                similarities.append(similarity_percent)
                camera_pairs.append((cam1_name, cam2_name))
                
                print(f"{cam1_name} vs {cam2_name}: {similarity_percent:.2f}%")
                
            except Exception as e:
                print(f"Error comparing {cam1_name} vs {cam2_name}: {e}")
    
    if similarities:
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        print(f"\n--- Summary ---")
        print(f"Average similarity between different cameras: {avg_similarity:.2f}%")
        print(f"Maximum similarity: {max_similarity:.2f}%")
        print(f"Minimum similarity: {min_similarity:.2f}%")
        
        # Check if results are reasonable
        if avg_similarity > 80:
            print("⚠️  WARNING: Average similarity is too high! Fingerprints may not be unique.")
        elif avg_similarity < 20:
            print("✅ Good: Low average similarity indicates unique fingerprints.")
        else:
            print("⚠️  CAUTION: Moderate similarity - may need further optimization.")

# === Compare same camera fingerprints (should be high similarity) ===
def compare_same_camera(fingerprints_dir):
    print(f"\n=== Testing same camera consistency in: {fingerprints_dir} ===")
    
    # This would require multiple fingerprints from the same camera
    # For now, just report that this test is not available
    print("Same camera consistency test not implemented (requires multiple fingerprints per camera)")

# === Main execution ===
if __name__ == "__main__":
    print("PRNU Fingerprint Comparison Tool")
    print("=" * 50)
    
    # Compare old fingerprints
    if os.path.exists(fingerprints_dir_old):
        compare_different_cameras(fingerprints_dir_old)
    else:
        print(f"Old fingerprints directory not found: {fingerprints_dir_old}")
    
    # Compare new fingerprints (if they exist)
    if os.path.exists(fingerprints_dir_new):
        compare_different_cameras(fingerprints_dir_new)
    else:
        print(f"New fingerprints directory not found: {fingerprints_dir_new}")
        print("Run extractionMain.py first to generate improved fingerprints")
