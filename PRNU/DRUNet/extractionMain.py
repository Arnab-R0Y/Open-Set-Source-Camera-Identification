import os
import cv2
import numpy as np
from tqdm import tqdm
from extract_PRNU import extract_multiple_aligned
from extract_PRNU import extract_patches
import multiprocessing
import time


if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_time = time.time()

    BASE_PATH = r'X:\Projects\Open set source camera identification'

    CAMERAS_ROOT = os.path.join(BASE_PATH, 'Images', 'Known Cameras')
    camera_folders = [f for f in os.listdir(CAMERAS_ROOT) if os.path.isdir(os.path.join(CAMERAS_ROOT, f))]

    for CAMERA_FOLDER in camera_folders:
        camera_start_time = time.time()
        print(f"\nStarting fingerprint extraction for: {CAMERA_FOLDER}")
        CAMERA_PATH = os.path.join(CAMERAS_ROOT, CAMERA_FOLDER)

        OUTPUT_DIR = os.path.join(BASE_PATH, 'Fingerprints_Laptop', CAMERA_FOLDER)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f'{CAMERA_FOLDER}_fingerprint.png')
        OUTPUT_NUMPY_PATH = os.path.join(OUTPUT_DIR, f'{CAMERA_FOLDER}_fingerprint.npy')
        if os.path.exists(OUTPUT_NUMPY_PATH):
            print(f"✅ Fingerprint already exists for {CAMERA_FOLDER}, skipping...")
            continue

        # Read image paths
        image_paths = [os.path.join(CAMERA_PATH, f) for f in os.listdir(CAMERA_PATH)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif'))]
        if not image_paths:
            raise FileNotFoundError(f"No images found in {CAMERA_PATH}")

        # Laptop-friendly settings
        MAX_IMAGES = 15  # Reasonable number for laptop
        image_paths = image_paths[:MAX_IMAGES]

        print(f"Processing {len(image_paths)} images for PRNU extraction...")

        # First, collect all patches from all images
        all_patches = []
        print("Loading all images and extracting patches...")
        
        for i, path in enumerate(image_paths, 1):
            try:
                if os.path.getsize(path) > 10 * 1024 * 1024:  # 10MB limit
                    print(f"Skipping large image: {path}")
                    continue

                im = cv2.imread(path)
                if im is None:
                    print(f"Could not read image: {path}")
                    continue
                
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                # Extract patches from this image
                patches = extract_patches(im, patch_size=512, stride=512)  # 512x512 patches
                patches = patches[:5]  # Take first 5 patches per image

                if not patches:
                    print(f"No patches extracted from {path}")
                    continue

                all_patches.extend(patches)
                
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        
        print(f"✅ Completed loading {len(image_paths)} images - Total patches collected: {len(all_patches)}")

        if not all_patches:
            print(f"No valid patches found for {CAMERA_FOLDER}")
            continue

        print(f"Total patches collected: {len(all_patches)}")

        # Now process patches in batches of 10
        PATCH_BATCH_SIZE = 10  # Process 10 patches at a time
        all_fingerprints = []
        
        for batch_start in range(0, len(all_patches), PATCH_BATCH_SIZE):
            batch_patches = all_patches[batch_start:batch_start + PATCH_BATCH_SIZE]
            
            batch_num = batch_start//PATCH_BATCH_SIZE + 1
            total_batches = (len(all_patches)-1)//PATCH_BATCH_SIZE + 1
            end_patch = min(batch_start + PATCH_BATCH_SIZE, len(all_patches))
            print(f"Processing batch {batch_num}/{total_batches} (Patches {batch_start+1}-{end_patch})")

            print(f"🛠️  Extracting PRNU for batch {batch_num} ({len(batch_patches)} patches)")
            
            # Extract PRNU from this batch of patches
            prnu = extract_multiple_aligned(batch_patches, levels=15, sigma=15, processes=4)
            
            if all_fingerprints and prnu.shape != all_fingerprints[0].shape:
                print(f"Skipping PRNU from batch {batch_num} due to shape mismatch: {prnu.shape}")
                continue
                
            all_fingerprints.append(prnu)
            print(f"✅ Batch {batch_num} completed ({len(batch_patches)} patches processed)")
            print()  # Add blank line for better formatting

        if not all_fingerprints:
            print(f"No valid fingerprints extracted for {CAMERA_FOLDER}")
            continue

        # Combine PRNUs from all batches
        print("Averaging fingerprints from all batches...")
        final_fingerprint = np.mean(all_fingerprints, axis=0)

        # Save outputs
        fingerprint_norm = cv2.normalize(final_fingerprint, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(OUTPUT_IMAGE_PATH, fingerprint_norm)
        np.save(OUTPUT_NUMPY_PATH, final_fingerprint)

        print(f"Saved fingerprint image at: {OUTPUT_IMAGE_PATH}")
        print(f"Saved fingerprint array at: {OUTPUT_NUMPY_PATH}")

        camera_end_time = time.time()
        camera_elapsed = camera_end_time - camera_start_time
        print(f"Time taken for {CAMERA_FOLDER}: {camera_elapsed:.2f} seconds")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Extraction script completed in {elapsed_time:.2f} seconds")
