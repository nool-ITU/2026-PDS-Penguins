import pandas as pd

# for bad images
with open("../data/img_delete.txt") as text:
    txt = text.read()
bad_list = txt.split(sep = "\n")
 
# for 2-3 lesions images
with open('../data/2-3 lesions.txt', 'r') as f:
    lesion_list = {line.strip() for line in f if line.strip()}

df = pd.read_csv("../data/metadata.csv")
df['2_3_lesions'] = df['img_id'].isin(lesion_list)
data_new = df[~df["img_id"].isin(bad_list)]

data_new.to_csv("../data/clean_data.csv", index=False)