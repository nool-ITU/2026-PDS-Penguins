from skimage.transform import rotate, resize
import numpy as np
from skimage import morphology 
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from math import ceil, floor
import cv2
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'
OUTPUT_FILE = 'extracted_features.csv'

# --- FEATURE EXTRACTION FUNCTIONS ---

def cut_mask(mask):
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)
    active_cols = np.where(col_sums != 0)[0]
    active_rows = np.where(row_sums != 0)[0]
    
    if len(active_cols) == 0 or len(active_rows) == 0:
        return mask, (0, 0, 0, 0)
        
    col_min, col_max = active_cols[0], active_cols[-1]
    row_min, row_max = active_rows[0], active_rows[-1]
    
    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]
    return cut_mask_, (row_min, row_max, col_min, col_max)

def asymmetry(mask):
    row_mid, col_mid = mask.shape[0] / 2, mask.shape[1] / 2
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]
    
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)
    
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)
    
    asymmetry_score = (np.sum(hori_xor_area) + np.sum(vert_xor_area)) / (np.sum(mask) * 2)
    return round(asymmetry_score, 4)

def mean_asymmetry(mask, rotations=5):
    asymmetry_scores = []
    for i in range(rotations):
        degrees = 90 * i / rotations
        rotated_mask = rotate(mask, degrees)
        cutted_mask, _ = cut_mask(rotated_mask > 0.5)
        asymmetry_scores.append(asymmetry(cutted_mask))
    return sum(asymmetry_scores) / len(asymmetry_scores)

def get_compactness(mask):
    area = np.sum(mask)
    if area == 0: return 0
    struct_el = morphology.disk(3)
    mask_eroded = morphology.erosion(mask, struct_el)
    perimeter = np.sum(mask.astype(int) - mask_eroded.astype(int))
    return (perimeter**2) / (4 * np.pi * area)

def convexity_score(mask):
    coords = np.transpose(np.nonzero(mask))
    if len(coords) < 3: return 0
    hull = ConvexHull(coords)
    lesion_area = np.count_nonzero(mask)
    return lesion_area / (hull.volume + hull.area)

def get_multicolor_rate(im, mask, n):
    im_small = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask_small = resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=False) > 0
    col_list = im_small[mask_small]
    if len(col_list) == 0: return 0
    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    centers = cluster.cluster_centers_
    dist = 0
    for i in range(len(centers)-1):
        dist = max(dist, np.linalg.norm(centers[i] - centers[i+1]))
    return dist

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        name_only = os.path.splitext(filename)[0]
        parts = name_only.split('_')
        
        mask_path = os.path.join(MASK_FOLDER, f"{name_only}_mask{os.path.splitext(filename)[1]}")
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None: return None
        mask_bool = mask > 127
        
        return {
            "patient_id": f"{parts[0]}_{parts[1]}",
            "lesion_id": parts[2],
            "Asymmetry": mean_asymmetry(mask_bool),
            "Compactness": get_compactness(mask_bool),
            "Convexity": convexity_score(mask_bool),
            "Multicolor": get_multicolor_rate(img, mask_bool, n=3)
        }
    except Exception:
        return None

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    total = len(files)
    print(f"Starting processing: {total} images found.")

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, f) for f in files]
        for i, future in enumerate(futures):
            res = future.result()
            if res: results.append(res)
            
            # Progress notification every 10 images or at the end
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Data saved to {OUTPUT_FILE}")