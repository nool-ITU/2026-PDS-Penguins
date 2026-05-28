from skimage.transform import rotate
import numpy as np
from skimage import morphology 
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from math import ceil, floor
from skimage.transform import resize
import cv2
from numpy import nan
from skimage.segmentation import slic
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from scipy.stats import entropy
from skimage import exposure

# A: Asymmetry
def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask to cut

    Returns:
        cut mask
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1], (row_min, row_max, col_min, col_max)

    return cut_mask_

def midpointGroup9(image):
    '''Find midpoint of image array.'''
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    '''Calculate asymmetry score between 0 and 1 from vertical and horizontal axis
    on a binary mask, 0 being complete symmetry, 1 being complete asymmetry,
    i.e. no pixels overlapping when folding mask on x- and y-axis

    Args:
        input mask

    Returns:
        Float between 0 and 1 indicating level of asymmetry.
    '''

    row_mid, col_mid = midpointGroup9(mask)

    # Split mask into halves hortizontally and vertically
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Flip one half for each axis
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # Use logical xor to find pixels where only one half is present
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    # Compute sums of total pixels and pixels in asymmetry areas
    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):

    """Rotate the mask n times and calculate asymmetry for each rotation.
    Returns a dictionary of asymmetry scores for each rotation angle."""
    
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask, _ = cut_mask(rotated_mask) 

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

def mean_asymmetry(mask, rotations = 30):

    """Compute mean asymmetry score by averaging rotation_asymmetry results.
    More reliable than single-direction asymmetry."""

    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score

# B: Border

def get_compactness(mask):

    """ Measures how round vs irregular a lesion is.
    Formula: perimeter^2 / (4 * pi * area)
    Higher value = less compact."""

    area = np.sum(mask)
    struct_el = morphology.disk(3)
    mask_eroded = morphology.erosion(mask, struct_el)
    perimeter = np.sum(mask.astype(int) - mask_eroded.astype(int))
    return perimeter**2 / (4 * np.pi * area)

def convexity_score(mask):

    """Calculate convexity score between 0 and 1,
    with 0 indicating a smoother border and 1 a more crooked border."""

    
    coords = np.transpose(np.nonzero(mask))
    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)
    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)
    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area
    
    convexity = lesion_area / convex_hull_area
    return convexity

# C: Color

def get_com_col(cluster, centroids):
    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    start = 0
    for percent, color in colors:
        if percent > 0.05:
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(
            rect,
            (int(start), 0),
            (int(end), 50),
            color.astype("uint8").tolist(),
            -1,
        )
        start = end
    return com_col_list

def get_multicolor_rate(im, mask, n):

    """Measure the maximum color difference inside a lesion using KMeans clustering.

    Args:
        Original RGB image of the lesion.
        Binary mask of the lesion.
        Number of color clusters to use in KMeans.

    Returns:
        float
        Higher value = more color variation in the lesion."""
    # mask = color.rgb2gray(mask)
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask = resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=False)
    mask= mask > 0
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j] * 256)

    if len(col_list) == 0:
        return ""

    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    if m <= 1:
        return ""

    for i in range(0, m - 1):
        for j in range(i + 1, m): 
            col_1 = com_col_list[i]
            col_2 = com_col_list[j]
            dist_list.append(
                np.sqrt(
                    (col_1[0] - col_2[0]) ** 2
                    + (col_1[1] - col_2[1]) ** 2
                    + (col_1[2] - col_2[2]) ** 2
                )
            )
    return np.max(dist_list)

def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    '''Get SLIC segments of a lesion.

Args:
    image to segment
    lesion area (True = lesion)
    number of segments (default 50)
    balance color vs position (default 0.1)

Returns:
    segmented lesion labels
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)

    return slic_segments

def get_hsv_means(image, slic_segments):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        original image
        SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    hsv_image = rgb2hsv(image)
    hsv_means = []
    for i in range(1, np.max(slic_segments)+1):

        mask = slic_segments == i

        #Get average HSV values from segment
        hue_mean = circmean(hsv_image[:, :, 0][mask], high=1, low=0)
        sat_mean = np.mean(hsv_image[:, :, 1][mask])
        val_mean = np.mean(hsv_image[:, :, 2][mask])
        hsv_means.append(np.array([hue_mean, sat_mean, val_mean]))

    return hsv_means

def hsv_var(image, slic_segments):
    '''Get variance of HSV means for each segment in
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''

    
    if len(np.unique(slic_segments)) <= 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    hsv_means = np.array(hsv_means)

    hue_var = circvar(hsv_means[:, 0], high=1, low=0)
    sat_var = np.nanvar(hsv_means[:, 1])
    val_var = np.nanvar(hsv_means[:, 2])

    return hue_var, sat_var, val_var

def measure_red_in_lesion(img_rgb, mask):
    mask_cut, coords = cut_mask(mask)
    row_min, row_max, col_min, col_max = coords
    img_cut = img_rgb[row_min:row_max+1, col_min:col_max+1]

    
    hsv_cut = cv2.cvtColor(img_cut, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 15, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 15, 40])
    upper_red2 = np.array([180, 255, 255])

    mask_r = cv2.inRange(hsv_cut, lower_red1, upper_red1) + cv2.inRange(hsv_cut, lower_red2, upper_red2)
    
    _, mask_bin = cv2.threshold(mask_cut.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask_r, mask_bin)

    lesion_area = cv2.countNonZero(mask_bin)
    red_pixels = cv2.countNonZero(final_mask)

    return (red_pixels / lesion_area) * 100 if lesion_area > 0 else 0

def measure_blue_in_lesion(img_rgb, mask):
    mask_cut, coords = cut_mask(mask)
    row_min, row_max, col_min, col_max = coords
    img_cut = img_rgb[row_min:row_max+1, col_min:col_max+1]

    hsv_cut = cv2.cvtColor(img_cut, cv2.COLOR_RGB2HSV)
    
    
    lower_blue = np.array([90, 10, 40])
    upper_blue = np.array([140, 255, 255])

    mask_b = cv2.inRange(hsv_cut, lower_blue, upper_blue)

    _, mask_bin = cv2.threshold(mask_cut.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask_b, mask_bin)

    lesion_area = cv2.countNonZero(mask_bin)
    blue_pixels = cv2.countNonZero(final_mask)

    return (blue_pixels / lesion_area) * 100 if lesion_area > 0 else 0

def get_average_red_intensity(img_rgb, mask):
    mask_cut, coords = cut_mask(mask)
    row_min, row_max, col_min, col_max = coords
    img_cut = img_rgb[row_min:row_max+1, col_min:col_max+1]

    _, mask_bin = cv2.threshold(mask_cut.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    
    # Since img_rgb is explicitly RGB, Red is index 0
    red_channel = img_cut[:, :, 0]
    red_pixels_in_lesion = red_channel[mask_bin > 0]

    return np.mean(red_pixels_in_lesion) if red_pixels_in_lesion.size > 0 else 0.0

def get_average_blue_intensity(img_rgb, mask):
    mask_cut, coords = cut_mask(mask)
    row_min, row_max, col_min, col_max = coords
    img_cut = img_rgb[row_min:row_max+1, col_min:col_max+1]

    _, mask_bin = cv2.threshold(mask_cut.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    
    # Since img_rgb is explicitly RGB, Blue is index 2
    blue_channel = img_cut[:, :, 2]
    blue_pixels_in_lesion = blue_channel[mask_bin > 0]

    return np.mean(blue_pixels_in_lesion) if blue_pixels_in_lesion.size > 0 else 0.0

def get_hue_entropy(image, mask, bins=32):
    hsv = rgb2hsv(image)
    hue_channel = hsv[:, :, 0]
    pixels_in_mask = hue_channel[mask > 0]
    if len(pixels_in_mask) == 0:
        return 0
    hist, _ = np.histogram(pixels_in_mask, bins=bins, range=(0, 1), density=True)
    return entropy(hist)

# hair feature

def hair_coverage_dark(img_gray):
    img_gray = cv2.medianBlur(img_gray, 9)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (45, 45))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)

    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = np.sum(hair_mask == 255)

    return hair_area / total_area


def hair_coverage_light(img_gray):
    img_gray = cv2.medianBlur(img_gray, 9)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 21))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    _, hair_mask = cv2.threshold(tophat, 75, 255, cv2.THRESH_BINARY)

    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = np.sum(hair_mask == 255)

    return hair_area / total_area


def get_hair_feature(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # contrast normalization
    p2, p98 = np.percentile(gray, (2, 98))
    gray = exposure.rescale_intensity(gray, in_range=(p2, p98))

    dark = hair_coverage_dark(gray)
    light = hair_coverage_light(gray)

    return round(dark + light, 4)


# DATA EXTRACTION & FILE SAVING


import os
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

IMAGE_FOLDER = '../data/shortcuts_imgs'
MASK_FOLDER = '../data/masks'
OUTPUT_FILE = '../data/extracted_features_extended.csv'

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        name_only, extension = os.path.splitext(filename)
        
        parts = name_only.split('_')
        if len(parts) < 3:
            return None

        p_id = f"{parts[0]}_{parts[1]}"
        l_id = parts[2]

        img_bgr = cv2.imread(filepath)

        original_path = filepath.replace("../data/shortcuts_imgs", "../data/imgs")
        img_bgr_original = cv2.imread(original_path)

        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(MASK_FOLDER, f"{name_only}_mask{extension}")
        if not os.path.exists(mask_path):
            return None
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask_bool = mask > 127

        if np.sum(mask_bool) == 0:
            return None
        
        if img_bgr_original is None:
            hair_feature = 0
        else:
            hair_feature = get_hair_feature(img_bgr_original)


        asym = mean_asymmetry(mask_bool, rotations=5)
        comp = get_compactness(mask_bool)
        conv = convexity_score(mask_bool)
        m_color = get_multicolor_rate(img, mask_bool, n=3)
        red = measure_red_in_lesion(img, mask_bool)
        blue = measure_blue_in_lesion(img, mask_bool)
        rgb_red = get_average_red_intensity(img, mask_bool)
        rgb_blue = get_average_blue_intensity(img, mask_bool)
        entropy_value= get_hue_entropy(img, mask_bool)
        slic_seg = slic_segmentation(img, mask_bool)
        h_v, s_v, v_v = hsv_var(img, slic_seg)
        hair_feature = get_hair_feature(img_bgr_original)

        return {
            "patient_id": p_id,
            "lesion_id": l_id,
            "filename": filename,
            "Asymmetry": asym,
            "Compactness": comp,
            "Convexity": conv,
            "Multicolor": m_color,
            "red_pixels": red,
            "blue_pixels": blue,
            "mean_red": rgb_red,
            "mean_blue": rgb_blue,
            "Entropy": entropy_value,
            "Hue_Var": h_v,
            "Sat_Var": s_v,
            "Val_Var": v_v,
            "hair_feature": hair_feature
        }
    except Exception as e:
        return None

if __name__ == '__main__':
    if not os.path.exists(IMAGE_FOLDER) or not os.path.exists(MASK_FOLDER):
        print(f"Error: Target folders '{IMAGE_FOLDER}' and/or '{MASK_FOLDER}' not found.")
        exit(1)

    files = [
        os.path.join(IMAGE_FOLDER, f) 
        for f in os.listdir(IMAGE_FOLDER) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    total_files = len(files)
    if total_files == 0:
        print(f"No valid images found inside the '{IMAGE_FOLDER}' directory.")
        exit(0)

    print(f"Processing {total_files} images using parallel process workers...")

    results = []
    processed_count = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, filepath): filepath for filepath in files}
        
        for future in as_completed(futures):
            processed_count += 1
            result = future.result()
            
            if result is not None:
                results.append(result)
            
            # Displays live progress inline
            print(f"\rProgress: {processed_count}/{total_files} files processed...", end="", flush=True)

    print("\nProcessing complete! Finalizing output data structure...")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! {len(results)} records successfully generated and saved to {OUTPUT_FILE}")
    else:
        print("Data processing ended, but no metrics were saved. Please check file formatting or mask profiles.")
