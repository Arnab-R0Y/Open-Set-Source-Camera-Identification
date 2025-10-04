"""
combine_noiseprint_prnu.py

Usage:
  - Put this file in your project (same root where Noiseprint.py lives or adjust import).
  - Call combine_image_and_prnu(image_path, prnu_path, out_prefix=..., fusion_mode='weighted', weight=0.5)
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional
# from Noiseprint import getNoiseprint   # your Noiseprint.py exports getNoiseprint
import warnings

import sys
sys.path.append(r"X:\Projects\Open set source camera identification")
sys.path.append(r"X:\Projects\Open set source camera identification\Noiseprint_pytorch")
from Noiseprint_pytorch.Noiseprint import getNoiseprint


# Optional: FLD helper uses numpy only (no sklearn)
def compute_dprnu(Wm: np.ndarray, Ki: np.ndarray) -> float:
    """DPRNU distance = 1 - NCC between Wm and Ki. Both inputs 2D floats."""
    w = Wm.astype(np.float64).ravel()
    k = Ki.astype(np.float64).ravel()
    # zero-mean as commonly done
    w = w - w.mean()
    k = k - k.mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(k) + 1e-12)
    ncc = (w @ k) / denom
    return float(1.0 - ncc)

def compute_dnp(phi_im: np.ndarray, R_i: np.ndarray) -> float:
    """DNP distance = MSE between phi_im and reference R_i"""
    phi = phi_im.astype(np.float64)
    R = R_i.astype(np.float64)
    return float(np.mean((phi - R) ** 2))

def load_prnu(prnu_path: str) -> np.ndarray:
    """Load PRNU residual from .npy or image file. Returns 2D float array."""
    if not os.path.exists(prnu_path):
        raise FileNotFoundError(prnu_path)
    ext = prnu_path.lower().split('.')[-1]
    if ext == 'npy':
        data = np.load(prnu_path)
    else:
        im = cv2.imread(prnu_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"cv2 failed to load {prnu_path}")
        # convert to grayscale/float
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        data = im.astype(np.float32)
    # convert to float32 and normalize to zero-mean scale-preserved
    data = data.astype(np.float32)
    return data

def align_and_match_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make two 2D arrays same size:
     - if shapes equal: return as-is
     - else: center-crop the larger to the smaller shape
     - if dimensions incompatible, resize b to a using cv2.resize (linear)
    Returns tuple (a2, b2)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    ha, wa = a.shape
    hb, wb = b.shape

    # if identical
    if ha == hb and wa == wb:
        return a, b

    # use center-crop to min size
    mh = min(ha, hb)
    mw = min(wa, wb)

    def crop_center(x, target_h, target_w):
        h, w = x.shape
        ys = (h - target_h) // 2
        xs = (w - target_w) // 2
        return x[ys:ys + target_h, xs:xs + target_w]

    a_c = crop_center(a, mh, mw)
    b_c = crop_center(b, mh, mw)
    return a_c, b_c

def normalize_to_minus1_1(x: np.ndarray) -> np.ndarray:
    """Scale array to [-1,1] robustly (min-max)."""
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    norm = (x - mn) / (mx - mn)
    return 2.0 * norm - 1.0

# FLD training helper (2-class). X shape (N,2), labels {0,1}
def train_fld(X: np.ndarray, labels: np.ndarray):
    X = np.asarray(X)
    labels = np.asarray(labels)
    assert X.ndim == 2 and X.shape[1] == 2
    X1 = X[labels == 1]
    X0 = X[labels == 0]
    mu1 = X1.mean(axis=0)
    mu0 = X0.mean(axis=0)
    S1 = np.cov(X1, rowvar=False)
    S0 = np.cov(X0, rowvar=False)
    S = S0 + S1
    # regularize a bit for numerical stability
    S += np.eye(2) * 1e-6
    w = np.linalg.solve(S, (mu1 - mu0))
    return w, mu0, mu1, S0, S1

def fld_score(w: np.ndarray, dprnu: float, dnp: float) -> float:
    return float(w.dot(np.array([dprnu, dnp])))

def combine_scores_weighted(dprnu: float, dnp: float, weight: float = 0.5) -> float:
    """Simple weighted combination; weight is for DPRNU (0..1)"""
    return float(weight * dprnu + (1.0 - weight) * dnp)

def zscore(values: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None):
    values = np.asarray(values, dtype=np.float64)
    if mean is None:
        mean = values.mean()
    if std is None:
        std = values.std(ddof=0) + 1e-12
    return (values - mean) / std

def pixelwise_fusion(noiseprint: np.ndarray, prnu: np.ndarray) -> np.ndarray:
    """Return fused image by normalizing both to [-1,1] and averaging."""
    n = normalize_to_minus1_1(noiseprint)
    p = normalize_to_minus1_1(prnu)
    # align shapes
    n_al, p_al = align_and_match_shape(n, p)
    fused = (n_al + p_al) / 2.0
    return fused

# -----------------------
# Main wrapper function
# -----------------------
def combine_image_and_prnu(image_path: str,
                           prnu_path: str,
                           out_prefix: Optional[str] = None,
                           fusion_mode: str = 'weighted',
                           weight: float = 0.5,
                           fld_train_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           calib_samples: Optional[np.ndarray] = None):
    """
    image_path: path to original image (Noiseprint.getNoiseprint will be used)
    prnu_path: path to PRNU residual (.npy or image)
    out_prefix: optional prefix for saved outputs
    fusion_mode: 'weighted' | 'zscore+weighted' | 'fld'
    weight: DPRNU weight for weighted fusion
    fld_train_pairs: optional training (X, labels) for FLD: X shape (N,2), labels shape (N,)
    calib_samples: optional calibration samples dict-like with keys 'dprnu' and 'dnp' arrays to compute z-score
    Returns dict with distances and score and paths saved.
    """
    if out_prefix is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base = out_prefix

    # 1) compute noiseprint
    print("Computing noiseprint ...")
    img_float, noiseprint_res = getNoiseprint(image_path)   # returns (img, res) per your function
    noiseprint_res = noiseprint_res.astype(np.float32)

    # 2) load PRNU residual (user said they already extracted PRNU and will provide link)
    print("Loading PRNU residual ...")
    prnu_res = load_prnu(prnu_path).astype(np.float32)

    # 3) align shapes
    n_al, p_al = align_and_match_shape(noiseprint_res, prnu_res)
    print(f"Aligned shapes: noiseprint {n_al.shape}, prnu {p_al.shape}")

    # 4) compute distances
    dprnu = compute_dprnu(n_al, p_al)   # note: paper computes between residual Wm (test residual) and Ki (reference)
    # but if you want DPRNU between PRNU_ref and test residual, pass accordingly. Here we treat both as same-sized residuals.
    dnpt = compute_dnp(n_al, p_al)     # phi_im vs R_i analog (we're using same arrays aligned)
    # If you have separate noiseprint reference R_i and test phi_im, substitute accordingly.

    print(f"DPRNU (1 - NCC): {dprnu:.8f}")
    print(f"DNP (MSE)     : {dnpt:.8f}")

    # 5) fusion
    score = None
    if fusion_mode == 'weighted':
        score = combine_scores_weighted(dprnu, dnpt, weight=weight)
        print(f"Weighted fusion score (w={weight}): {score:.6f}")

    elif fusion_mode == 'zscore+weighted':
        if calib_samples is None:
            warnings.warn("No calibration samples given — falling back to simple weighted")
            score = combine_scores_weighted(dprnu, dnpt, weight=weight)
        else:
            # calib_samples expected dict-like: {'dprnu':array, 'dnp':array}
            mu_dprnu = np.mean(calib_samples['dprnu'])
            sd_dprnu = np.std(calib_samples['dprnu']) + 1e-12
            mu_dnp = np.mean(calib_samples['dnp'])
            sd_dnp = np.std(calib_samples['dnp']) + 1e-12
            z1 = (dprnu - mu_dprnu) / sd_dprnu
            z2 = (dnpt - mu_dnp) / sd_dnp
            # combine z-scores (equal weight)
            score = float(0.5 * z1 + 0.5 * z2)
            print(f"Z-score fusion: {score:.6f} (z_dprnu={z1:.3f}, z_dnp={z2:.3f})")

    elif fusion_mode == 'fld':
        if fld_train_pairs is None:
            raise ValueError("fld mode requires fld_train_pairs=(X, labels)")
        X, labels = fld_train_pairs
        w, *_ = train_fld(X, labels)
        score = fld_score(w, dprnu, dnpt)
        print(f"FLD score: {score:.6f}")

    else:
        raise ValueError("Unknown fusion_mode")

    # 6) pixel-wise fused residual (visual)
    fused_image = pixelwise_fusion(noiseprint_res, prnu_res)
    # save outputs
    out_dir = os.path.join(".", "combined_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_comb_png = os.path.join(out_dir, base + "_combined_visual.png")
    out_np_noiseprint = os.path.join(out_dir, base + "_noiseprint.npy")
    out_np_prnu = os.path.join(out_dir, base + "_prnu.npy")
    out_np_comb = os.path.join(out_dir, base + "_combined.npy")

    # normalize fused image to uint8 for saving
    fused_uint8 = ((fused_image - fused_image.min()) / (fused_image.max() - fused_image.min() + 1e-12) * 255.0).astype(np.uint8)
    cv2.imwrite(out_comb_png, fused_uint8)
    np.save(out_np_noiseprint, noiseprint_res)
    np.save(out_np_prnu, prnu_res)
    np.save(out_np_comb, fused_image)

    print(f"Saved visual combined image: {out_comb_png}")
    print(f"Saved residual arrays: {out_np_noiseprint}, {out_np_prnu}, {out_np_comb}")

    return {
        'dprnu': dprnu,
        'dnp': dnpt,
        'fusion_score': score,
        'paths': {
            'combined_png': out_comb_png,
            'noiseprint_npy': out_np_noiseprint,
            'prnu_npy': out_np_prnu,
            'combined_npy': out_np_comb
        }
    }

# --------------------
# Example usage
# --------------------
def example_usage():
    # change these to your actual files
    image_path = r"X:\Projects\Open set source camera identification\Images\Known Cameras\Agfa_DC-504_0\Agfa_DC-504_0_1.JPG"
    prnu_path = r"X:\Projects\Open set source camera identification\output_images\Agfa_DC-504_0_1_prnu_recon.png"

    # simple weighted fusion
    res = combine_image_and_prnu(image_path, prnu_path, fusion_mode='weighted', weight=0.5)
    print(res)

    # Example: using zscore fusion with calibration (if you have calibration arrays)
    # calib = {'dprnu': np.array([...]), 'dnp': np.array([...])}
    # res2 = combine_image_and_prnu(image_path, prnu_path, fusion_mode='zscore+weighted', calib_samples=calib)

    # Example: FLD training if you have training data X (N,2) and labels (N,)
    # X_train = np.vstack([dprnu_array, dnp_array]).T
    # labels = ...
    # res3 = combine_image_and_prnu(image_path, prnu_path, fusion_mode='fld', fld_train_pairs=(X_train, labels))

if __name__ == "__main__":
    example_usage()
