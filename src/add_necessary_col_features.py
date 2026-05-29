import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/clean_data_with_splits.csv")


features_b = pd.read_csv("../data/extracted_features_baseline.csv")
features_b["group_id"]=df["group_id"]
features_b["Class"]=df["Class"]
features_b["2_3_lesions"]=df["2_3_lesions"]


train_mask = features_b["group_id"].isin(["a", "b", "c", "d"])

one_lesion_mask = features_b["2_3_lesions"] == False

cols = ['Asymmetry', 'Compactness', 'Convexity']

for col in cols:
    avg_val_train = features_b.loc[train_mask & one_lesion_mask, col].mean()
    features_b.loc[features_b['2_3_lesions'] == True, col] = avg_val_train
features_b.to_csv('../data/extracted_features_baseline_mean.csv', index=False)


features_e = pd.read_csv("../data/extracted_features_extended.csv")

features_e["2_3_lesions"]=df["2_3_lesions"]
features_e["group_id"]=df["group_id"]
features_e["Class"]=df["Class"]

train_mask = features_e["group_id"].isin(["a", "b", "c", "d"])

one_lesion_mask = features_e["2_3_lesions"] == False

cols = ['Asymmetry', 'Compactness', 'Convexity']

for col in cols:
    avg_val_train = features_e.loc[train_mask & one_lesion_mask, col].mean()
    features_e.loc[features_e['2_3_lesions'] == True, col] = avg_val_train
features_e.to_csv('../data/extracted_features_extended_mean.csv', index=False)
