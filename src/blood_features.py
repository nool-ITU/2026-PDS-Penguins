import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("../data/clean_data_with_splits.csv")
# df = df[df["group_id"].isin(['a', 'b', 'c', 'd'])]

def blood_detector(image):
    img = cv2.imread(image)

    B, G, R = cv2.split(img)
    R, G, B = R.astype(float), G.astype(float), B.astype(float)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    red_ratio = R / (G + B + 1)
    score = R - 0.65 * G - 0.45 * B
    darkness = (R + G + B) / 3
    red_dom = R - G

    brown_suppression = ~((H > 15) & (H < 35) & (S < 80) & (V > 80))

    mask = (
        (red_ratio > 1.20) &
        (score > 50) &
        (darkness < 160) &
        (red_dom > 30) &
        (S > 60) &
        brown_suppression
    ).astype(np.uint8) * 255

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            filtered[labels == i] = 255

    blood_pixels = np.sum(filtered > 0)
    has_blood = blood_pixels > 400
    return has_blood, blood_pixels

blood_detected = []
pixels = []
for i in df['img_id']:
    path = f"../data/imgs/{i}"
    a, b = blood_detector(path)
    blood_detected.append(str(bool(a)))
    pixels.append(int(b))
# Add the new features to existing features csvs
base_mean = pd.read_csv("../data/extracted_features_baseline_mean.csv")
extended_mean = pd.read_csv("../data/extracted_features_extended_mean.csv")

df["blood_detected"] = blood_detected
df["blood_pixels"] = pixels

# drop old columns if any exist
cols_to_drop = ["blood_detected", "blood_pixels"]
base_mean = base_mean.drop(columns=[c for c in cols_to_drop if c in base_mean], errors="ignore")
extended_mean = extended_mean.drop(columns=[c for c in cols_to_drop if c in extended_mean], errors="ignore")

# fix mismatch in lengths
df_filtered = df[["img_id", "blood_detected", "blood_pixels"]]
base_mean = base_mean.merge(df_filtered, left_on="filename", right_on="img_id", how="left")
extended_mean = extended_mean.merge( df_filtered, left_on="filename", right_on="img_id", how="left")
base_mean = base_mean.drop(columns=["img_id"])
extended_mean = extended_mean.drop(columns=["img_id"])

base_mean.to_csv("../data/extracted_features_baseline_mean_blood.csv", index=False)
extended_mean.to_csv("../data/extracted_features_extended_mean_blood.csv", index=False)