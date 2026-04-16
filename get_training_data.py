import pandas as pd
import shutil
from pathlib import Path

#Define paths to data and the group ID to get the data for
GROUP_ID = ["b", "c", "d", "e"]
PATH_RAW = Path("./data")
PATH_PROCESSED = Path("./data/training_data1")
PATH_IMGS = PATH_PROCESSED/"imgs/"
PATH_MASKS = PATH_PROCESSED/"masks/"


#Create the subfolders to save the data into
PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
PATH_IMGS.mkdir(parents=True, exist_ok=True)
PATH_MASKS.mkdir(parents=True, exist_ok=True)

#Load the csv and filter it to only keep the samples assigned to the specified group
df_labels = pd.read_csv(f"{PATH_RAW}/data_with_splits.csv")
df_labels_group = df_labels[df_labels["group_id"]==GROUP_ID]

#For each image assigned to the group, copy it and its mask to the group_data folder
for img_path in df_labels_group["img_id"]:
    try:
        shutil.copyfile(f"{PATH_RAW}/imgs/{img_path}", f"{PATH_IMGS}/{img_path}")
        shutil.copyfile(f"{PATH_RAW}/masks/{img_path.replace('.png','_mask.png')}", f"{PATH_MASKS}/{img_path.replace('.png','_mask.png')}")
    except:
        print(f"PROBLEM WITH {img_path}, please check that both the image and associated mask are present in the folders")

#Save the filtered metadata files in the group_data folder
df_labels_group.to_csv(f"{PATH_PROCESSED}/metadata.csv")