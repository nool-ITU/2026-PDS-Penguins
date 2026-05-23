import pandas as pd

# Load your existing CSV
df = pd.read_csv('data_with_splits.csv')

# Load the file with the 2-3 lesions list
with open('2-3 lesions.txt', 'r') as f:
    lesion_list = {line.strip() for line in f if line.strip()}

# Add the column
df['2_3_lesions'] = df['img_id'].isin(lesion_list)

# Save the update
df.to_csv('data_with_splits_updated.csv', index=False)


