import csv
from pathlib import Path

# Automatically detects the directory where this script is saved
SCRIPT_DIR = Path(__file__).resolve().parent

# Configuration
csv_file = SCRIPT_DIR / 'clean_data_with_splits.csv'
img_dir = SCRIPT_DIR / 'imgs_all'
mask_dir = SCRIPT_DIR / 'masks_all'
csv_column = 'img_id'

# 1. Read the CSV to build independent keep-sets for images and masks
keep_images = set()
keep_masks = set()

with open(csv_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row[csv_column].strip()
        if img_name:
            keep_images.add(img_name)
            
            # Extracts 'PAT_161_250_197' and turns it into 'PAT_161_250_197_mask.png'
            base_name = Path(img_name).stem
            mask_name = f"{base_name}_mask.png"
            keep_masks.add(mask_name)

print(f"Loaded {len(keep_images)} target filenames from CSV.\n")

# 2. Function to scan and purge a directory based on its specific keep-set
def clean_directory(target_dir, keep_set, label):
    if not target_dir.exists():
        print(f"Skipping {label}: Directory does not exist at {target_dir}")
        return

    deleted_count = 0
    kept_count = 0

    print(f"--- CLEANING {label.upper()} FOLDER ---")
    for file_path in target_dir.iterdir():
        if file_path.is_file():
            # Skip hidden system files like .DS_Store
            if file_path.name.startswith('.'):
                continue
                
            if file_path.name in keep_set:
                kept_count += 1
            else:
                try:
                    file_path.unlink()  # Deletes permanently
                    print(f"Deleted unlisted {label}: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")

    print(f"Result for {label}: Kept {kept_count} files, Deleted {deleted_count} files.\n")

# 3. Execute cleanup for both targets
clean_directory(img_dir, keep_images, "images")
clean_directory(mask_dir, keep_masks, "masks")

print("--- CLEANUP PROCESS COMPLETE ---")
