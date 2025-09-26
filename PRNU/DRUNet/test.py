import os
import cv2
import numpy as np
from tqdm import tqdm
from extract_PRNU import extract_multiple_aligned
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

        OUTPUT_DIR = os.path.join(BASE_PATH, 'Fingerprints', CAMERA_FOLDER)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f'{CAMERA_FOLDER}_fingerprint.png')
        OUTPUT_NUMPY_PATH = os.path.join(OUTPUT_DIR, f'{CAMERA_FOLDER}_fingerprint.npy')

        # Read image paths
        image_paths = [os.path.join(CAMERA_PATH, f) for f in os.listdir(CAMERA_PATH)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif'))]
        if not image_paths:
            raise FileNotFoundError(f"No images found in {CAMERA_PATH}")

        # Parameters
        BATCH_SIZE = 20
        all_fingerprints = []
        total_images = len(image_paths)

        print(f"Processing in batches of {BATCH_SIZE} (total: {total_images} images)")

        for i in range(0, total_images, BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            images = []

            for path in batch_paths:
                try:
                    if os.path.getsize(path) > 15 * 1024 * 1024:
                        print(f"Skipping large image: {path}")
                        continue

                    im = cv2.imread(path)
                    if im is None:
                        print(f"Could not read image: {path}")
                        continue
                    
                    # Resize to 512x512
                    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)

                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    images.append(im)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

            if not images:
                print(f"Skipping batch {i}–{i + BATCH_SIZE}: No valid images")
                continue

            print(f"🛠️  Extracting PRNU for batch {i}–{i + len(images)}")

            prnu = extract_multiple_aligned(images, levels=4, sigma=5, processes=4)
            
            if all_fingerprints and prnu.shape != all_fingerprints[0].shape:
                print(f"Skipping PRNU from batch {i}–{i + len(images)} due to shape mismatch: {prnu.shape}")
                continue
            all_fingerprints.append(prnu)

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
    print(f"Script completed in {elapsed_time:.2f} seconds")
