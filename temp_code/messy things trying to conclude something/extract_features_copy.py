import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from math import ceil, floor

# Import dalle tue librerie specifiche
from skimage.transform import rotate, resize
from skimage import morphology
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from skimage.segmentation import slic

# --- CONFIGURAZIONE ---
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'
OUTPUT_FILE = 'extracted_features.csv'
RESIZE_DIM = 512  # Ridimensiona a 512px per velocizzare drasticamente

# --- FUNZIONI DI CALCOLO (OTTIMIZZATE) ---

def cut_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return mask
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return mask[rmin:rmax+1, cmin:cmax+1]

def asymmetry(mask):
    if np.sum(mask) == 0: return 0
    h, w = mask.shape
    row_mid, col_mid = h / 2, w / 2

    upper = mask[:ceil(row_mid), :]
    lower = np.flip(mask[floor(row_mid):, :], axis=0)
    left = mask[:, :ceil(col_mid)]
    right = np.flip(mask[:, floor(col_mid):], axis=1)

    # Padding se le dimensioni non coincidono perfettamente dopo il flip
    def sync_shape(img1, img2):
        r = max(img1.shape[0], img2.shape[0])
        c = max(img1.shape[1], img2.shape[1])
        p1 = np.zeros((r, c), dtype=bool)
        p2 = np.zeros((r, c), dtype=bool)
        p1[:img1.shape[0], :img1.shape[1]] = img1
        p2[:img2.shape[0], :img2.shape[1]] = img2
        return p1, p2

    u, l = sync_shape(upper, lower)
    le, ri = sync_shape(left, right)

    hori_asym = np.sum(np.logical_xor(u, l))
    vert_asym = np.sum(np.logical_xor(le, ri))
    
    return round((hori_asym + vert_asym) / (np.sum(mask) * 2), 4)

def mean_asymmetry(mask, rotations=5):
    scores = []
    for i in range(rotations):
        deg = 90 * i / rotations
        # Usiamo un resize più aggressivo per l'asimmetria per velocità
        rot = rotate(mask.astype(float), deg) > 0.5
        scores.append(asymmetry(cut_mask(rot)))
    return np.mean(scores)

def get_compactness(mask):
    area = np.sum(mask)
    if area == 0: return 0
    eroded = morphology.binary_erosion(mask, morphology.disk(3))
    perimeter = np.sum(mask ^ eroded)
    return (perimeter**2) / (4 * np.pi * area)

def convexity_score(mask):
    coords = np.transpose(np.nonzero(mask))
    if len(coords) < 3: return 0
    hull = ConvexHull(coords)
    return np.count_nonzero(mask) / (hull.volume + hull.area)

def get_multicolor_rate(im, mask, n=3):
    # Lavoriamo su una versione piccola per KMeans
    small_im = resize(im, (100, 100), anti_aliasing=True)
    small_mask = resize(mask, (100, 100), anti_aliasing=False) > 0.5
    
    pixels = small_im[small_mask]
    if len(pixels) < n: return np.nan
    
    kmeans = KMeans(n_clusters=n, n_init=5).fit(pixels)
    centers = kmeans.cluster_centers_
    
    # Calcolo distanze euclidee tra i centri dei colori
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    return np.max(dists) if dists else 0

def get_hsv_features(image, mask):
    # Segmentazione SLIC
    segments = slic(image, n_segments=20, compactness=0.1, mask=mask, start_label=1, channel_axis=2)
    hsv = rgb2hsv(image)
    h_means, s_means, v_means = [], [], []
    
    for i in np.unique(segments):
        if i == 0: continue
        m = segments == i
        h_means.append(circmean(hsv[:,:,0][m], high=1, low=0))
        s_means.append(np.mean(hsv[:,:,1][m]))
        v_means.append(np.mean(hsv[:,:,2][m]))
    
    if not h_means: return 0, 0, 0
    return circvar(h_means), np.var(s_means), np.var(v_means)

# --- PROCESSO PRINCIPALE ---

def process_file(filepath):
    try:
        fname = os.path.basename(filepath)
        name_only, ext = os.path.splitext(fname)
        
        # Caricamento e ridimensionamento (Fondamentale per la velocità)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None: return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
        
        mask_path = os.path.join(MASK_FOLDER, f"{name_only}_mask{ext}")
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None: return None
        mask = cv2.resize(mask_gray, (RESIZE_DIM, RESIZE_DIM)) > 127

        # Estrazione Feature
        h_v, s_v, v_v = get_hsv_features(img, mask)
        
        # Parsing ID (Esempio: PAT_15_1001_749)
        parts = name_only.split('_')
        p_id = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else "unknown"
        l_id = parts[2] if len(parts) > 2 else "unknown"

        return {
            "patient_id": p_id,
            "lesion_id": l_id,
            "filename": fname,
            "Asymmetry": mean_asymmetry(mask, rotations=5),
            "Compactness": get_compactness(mask),
            "Convexity": convexity_score(mask),
            "Multicolor": get_multicolor_rate(img, mask),
            "Hue_Var": h_v, "Sat_Var": s_v, "Val_Var": v_v
        }
    except Exception as e:
        return None

if __name__ == '__main__':
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Cartella {IMAGE_FOLDER} non trovata!")
    else:
        files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"🔄 Avvio elaborazione di {len(files)} immagini...")
        results = []

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            
            # Barra di progresso TQDM
            for future in tqdm(as_completed(futures), total=len(files), desc="Analisi Lesioni"):
                res = future.result()
                if res:
                    results.append(res)

        if results:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ Finito! Salvati {len(results)} record in '{OUTPUT_FILE}'")
        else:
            print("\n❌ Errore: Nessun dato estratto. Controlla che i nomi delle maschere siano corretti.")