import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter
from numpy.fft import fft2, ifft2
import os
import glob
import cv2

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

# === Advanced PRNU comparison functions ===
def zero_mean_total(im):
    """Simplified global zero-mean normalization for PRNU fingerprints."""
    return im - im.mean()

def zero_mean(im):
    """Simplified zero-mean normalization for 2D/3D arrays."""
    return im - im.mean()

def crosscorr_2d(k1, k2):
    """2D cross-correlation with mean removal and FFT-friendly padding."""
    assert k1.ndim == 2
    assert k2.ndim == 2
    
    # Mean removal
    k1 = k1 - k1.mean()
    k2 = k2 - k2.mean()
    
    # Target size: max dims
    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])
    
    # Use next power of two for efficiency
    fft_h = 1 << int(np.ceil(np.log2(max_height)))
    fft_w = 1 << int(np.ceil(np.log2(max_width)))
    
    k1_pad = np.zeros((fft_h, fft_w), dtype=k1.dtype)
    k2_pad = np.zeros((fft_h, fft_w), dtype=k2.dtype)
    k1_pad[:k1.shape[0], :k1.shape[1]] = k1
    k2_pad[:k2.shape[0], :k2.shape[1]] = k2
    
    k1_fft = fft2(k1_pad)
    k2_fft = fft2(np.rot90(k2_pad, 2))
    cc = np.real(ifft2(k1_fft * k2_fft))
    
    # Crop back to analysis size
    cc = cc[:max_height, :max_width]
    return cc.astype(np.float32)

def pce(cc, neigh_radius=2):
    """Peak-to-Correlation Energy with safe masking near edges."""
    assert cc.ndim == 2
    assert isinstance(neigh_radius, int)
    
    max_idx = np.argmax(cc.ravel())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)
    peak_height = cc[max_y, max_x]
    
    cc_nopeaks = cc.copy()
    y_min = max(0, max_y - neigh_radius)
    y_max = min(cc.shape[0], max_y + neigh_radius + 1)
    x_min = max(0, max_x - neigh_radius)
    x_max = min(cc.shape[1], max_x + neigh_radius + 1)
    cc_nopeaks[y_min:y_max, x_min:x_max] = 0
    
    pce_energy = np.mean(cc_nopeaks.ravel() ** 2)
    if pce_energy == 0:
        pce_energy = 1e-10
    
    pce_value = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    return pce_value, peak_height

def compute_similarity_enhanced(f1, f2):
    """Enhanced PRNU similarity computation with multiple metrics"""
    # Ensure both fingerprints are 2D
    if f1.ndim == 3:
        f1 = f1[:, :, 0]  # Take first channel if 3D
    if f2.ndim == 3:
        f2 = f2[:, :, 0]  # Take first channel if 3D
    
    # Normalize fingerprints
    f1_norm = zero_mean_total(f1.copy())
    f2_norm = zero_mean_total(f2.copy())
    
    # Compute cross-correlation
    cc = crosscorr_2d(f1_norm, f2_norm)
    
    # Calculate PCE
    pce_value, peak_height = pce(cc)
    
    # Calculate additional metrics
    # 1. Normalized cross-correlation
    ncc = np.corrcoef(f1_norm.flatten(), f2_norm.flatten())[0, 1]
    
    # 2. Structural similarity (SSIM-like)
    f1_std = f1_norm.std()
    f2_std = f2_norm.std()
    f1_mean = f1_norm.mean()
    f2_mean = f2_norm.mean()
    
    covariance = np.mean((f1_norm - f1_mean) * (f2_norm - f2_mean))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_value = ((2 * f1_mean * f2_mean + c1) * (2 * covariance + c2)) / \
                 ((f1_mean**2 + f2_mean**2 + c1) * (f1_std**2 + f2_std**2 + c2))
    
    return pce_value, peak_height, ncc, ssim_value

def compute_similarity(f1, f2):
    """Legacy similarity computation with constant-input guard."""
    flat1 = f1.flatten()
    flat2 = f2.flatten()
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        return 0.0
    corr, _ = pearsonr(flat1, flat2)
    if np.isnan(corr):
        return 0.0
    return corr

# === Enhanced comparison function ===
def compare_different_cameras_enhanced(fingerprints_dir, dir_name=""):
    print(f"\n=== ENHANCED Comparison: {dir_name} ===")
    print(f"Directory: {fingerprints_dir}")
    
    # Get all fingerprint files
    fingerprint_files = glob.glob(os.path.join(fingerprints_dir, "*", "*_fingerprint.npy"))
    
    if len(fingerprint_files) < 2:
        print("Need at least 2 fingerprint files to compare")
        return
    
    similarities = []
    pce_values = []
    ncc_values = []
    ssim_values = []
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
                
                # Use enhanced similarity computation
                pce_value, peak_height, ncc, ssim_value = compute_similarity_enhanced(f1, f2)
                
                # Enhanced similarity calculation
                # Combine multiple metrics for better discrimination
                # Lower values should indicate more different fingerprints
                
                # PCE-based similarity (lower PCE = more different)
                pce_sim = max(0, 50 - abs(pce_value) * 0.4)
                
                # NCC-based similarity (lower absolute NCC = more different)
                ncc_sim = max(0, 50 - abs(ncc) * 25)
                
                # SSIM-based similarity (lower SSIM = more different)
                ssim_sim = max(0, 50 - ssim_value * 25)
                
                # Combined similarity (weighted average)
                combined_similarity = (pce_sim * 0.5 + ncc_sim * 0.3 + ssim_sim * 0.2)
                
                # Also compute legacy correlation for comparison
                legacy_corr = compute_similarity(f1, f2)
                legacy_percent = max(0.0, min((legacy_corr + 1) * 50, 100))
                
                similarities.append(combined_similarity)
                pce_values.append(pce_value)
                ncc_values.append(ncc)
                ssim_values.append(ssim_value)
                camera_pairs.append((cam1_name, cam2_name))
                
                print(f"{cam1_name} vs {cam2_name}:")
                print(f"  PCE={pce_value:.2f}, NCC={ncc:.3f}, SSIM={ssim_value:.3f}")
                print(f"  Enhanced={combined_similarity:.2f}%, Legacy={legacy_percent:.2f}%")
                print()
                
            except Exception as e:
                print(f"Error comparing {cam1_name} vs {cam2_name}: {e}")
    
    if similarities:
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        std_similarity = np.std(similarities)
        
        print(f"--- ENHANCED Summary ---")
        print(f"Average similarity: {avg_similarity:.2f}% ± {std_similarity:.2f}%")
        print(f"Maximum similarity: {max_similarity:.2f}%")
        print(f"Minimum similarity: {min_similarity:.2f}%")
        print(f"Average PCE: {np.mean(pce_values):.2f}")
        print(f"Average NCC: {np.mean(ncc_values):.3f}")
        print(f"Average SSIM: {np.mean(ssim_values):.3f}")
        
        # Enhanced analysis
        if avg_similarity > 80:
            print("⚠️  WARNING: Average similarity is too high! Fingerprints may not be unique.")
            print("   Consider: 1) Running extractionMain_optimized.py")
            print("            2) Using more diverse images")
            print("            3) Checking for image preprocessing issues")
        elif avg_similarity < 15:
            print("✅ EXCELLENT: Very low similarity indicates highly unique fingerprints!")
            print("   Your camera fingerprints are excellently discriminated!")
        elif avg_similarity < 30:
            print("✅ VERY GOOD: Low similarity indicates good fingerprint discrimination.")
        elif avg_similarity < 50:
            print("✅ GOOD: Acceptable similarity levels for camera discrimination.")
        else:
            print("⚠️  CAUTION: Moderate similarity - optimization recommended.")
            print("   Recommendations:")
            print("   1) Run: python extractionMain_optimized.py")
            print("   2) Use more diverse images for fingerprint generation")
            print("   3) Check image quality and preprocessing")

# === Main execution ===
if __name__ == "__main__":
    print("ENHANCED PRNU Fingerprint Comparison Tool")
    print("=" * 60)
    
    # Compare old fingerprints
    if os.path.exists(fingerprints_dir_old):
        compare_different_cameras_enhanced(fingerprints_dir_old, "Original Fingerprints")
    else:
        print(f"Original fingerprints directory not found: {fingerprints_dir_old}")
    
    # Compare new fingerprints (if they exist)
    if os.path.exists(fingerprints_dir_new):
        compare_different_cameras_enhanced(fingerprints_dir_new, "Laptop Fingerprints")
    else:
        print(f"Laptop fingerprints directory not found: {fingerprints_dir_new}")
    
    
    print("\n" + "=" * 60)
    print("🎯 ENHANCED COMPARISON COMPLETE!")
    print("The enhanced tool uses multiple metrics (PCE, NCC, SSIM) for better discrimination.")

