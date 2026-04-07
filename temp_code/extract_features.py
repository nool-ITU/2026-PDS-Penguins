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
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


# A: Asymmetry
def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask
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

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

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
        mask (numpy.ndarray): input mask

    Returns:
        asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.
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

    # Calculate asymmetry score
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):

    """Rotate the mask n times and calculate asymmetry for each rotation.
    Returns a dictionary of asymmetry scores for each rotation angle."""
    
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

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

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))
    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)
    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)
    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area
    # Compute convexity as ratio of lesion area to convex hull
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
        if percent > 0.08:
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
        image (numpy.ndarray): Original RGB image of the lesion.
        mask (numpy.ndarray): Binary mask of the lesion.
        n (int): Number of color clusters to use in KMeans.

    Returns:
        float: Maximum color difference between cluster centers.
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
        j = i + 1
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
    image (np.ndarray): image to segment
    mask (np.ndarray): lesion area (True = lesion)
    n_segments (int): number of segments (default 50)
    compactness (float): balance color vs position (default 0.1)

Returns:
    np.ndarray: segmented lesion labels
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
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

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

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) <= 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    hsv_means = np.array(hsv_means)

    hue_var = circvar(hsv_means[:, 0], high=1, low=0)
    sat_var = np.nanvar(hsv_means[:, 1])
    val_var = np.nanvar(hsv_means[:, 2])

    return hue_var, sat_var, val_var

# DATA EXTRACTION & FILE SAVING


import os
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

IMAGE_FOLDER = 'images'
MASK_FOLDER = 'masks'
OUTPUT_FILE = 'extracted_features.csv'


def process_file(filename):
    try:
        # 1. Parse IDs (e.g., PAT_15_1001_749.png)
        name_only = os.path.splitext(filename)[0]
        parts = name_only.split('_')
        # Combined PAT and Number for patient_id, third part for lesion_id
        p_id = f"{parts[0]}_{parts[1]}"
        l_id = parts[2]

        # 2. Load Image and Mask
        img = cv2.cvtColor(cv2.imread(f'images/{filename}'), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f'masks/{name_only}_mask{os.path.splitext(filename)[1]}', cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            return None
        mask_bool = mask > 127

        # 3. Call Harini's functions (FIXED: Added n=3)
        asym = mean_asymmetry(mask_bool, rotations=5)
        comp = get_compactness(mask_bool)
        conv = convexity_score(mask_bool)
        m_color = get_multicolor_rate(img, mask_bool, n=3) # Added n=3 here
        
        # HSV Variance
        h_v, s_v, v_v = hsv_var(img, slic_segmentation(img, mask_bool))

        return {
            "patient_id": p_id,
            "lesion_id": l_id,
            "filename": filename,
            "Asymmetry": asym,
            "Compactness": comp,
            "Convexity": conv,
            "Multicolor": m_color,
            "Hue_Var": h_v, "Sat_Var": s_v, "Val_Var": v_v
        }
    except Exception as e:
        return None

if __name__ == '__main__':
    # Gather files
    files = [f for f in os.listdir('images') if f.lower().endswith(('.png', '.jpg'))]
    print(f"Processing {len(files)} images...")

    # Faster parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, files, chunksize=10))

    # Save to CSV
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv('extracted_features.csv', index=False)
    print("Success! File saved as extracted_features.csv")