import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import random

df = pd.read_csv("../data/clean_data.csv")        
df = df.sample(frac=1, random_state=40).reset_index(drop=True)  

df['combined_y'] = df["diagnostic"].astype(str) + "_" + df["2_3_lesions"].astype(str)



sgkf = StratifiedGroupKFold(n_splits=5)
sgkf_generator = sgkf.split(X=df, y=df['combined_y'], groups=df['patient_id'])


all_splits = [fold for fold in sgkf_generator]



split_0 = all_splits[0]
split_1 = all_splits[1]
split_2 = all_splits[2]
split_3 = all_splits[3]
split_4 = all_splits[4]

df.loc[split_0[1], "group_id"] = "a"
df.loc[split_1[1], "group_id"] = "b"
df.loc[split_2[1], "group_id"] = "c"
df.loc[split_3[1], "group_id"] = "d"
df.loc[split_4[1], "group_id"] = "e"

df.to_csv("../data/clean_data_with_splits.csv", index=False)