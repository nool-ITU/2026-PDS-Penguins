import csv
import shutil
from pathlib import Path

# Configuration
csv_file = '../../data/clean_data_with_splits.csv'        # Path to your CSV file
csv_column = 'img_id'           # Column header name in your CSV
img_src_dir = Path('../../data/imgs')    # Source for images
img_dst_dir = Path('../../data/good_imgs')        # Destination for images
mask_src_dir = Path('../../data/masks')   # Source for masks
mask_dst_dir = Path('../../data/good_masks')      # Destination for masks

def move_files():
    # 1. Create destination directories if they don't exist
    img_dst_dir.mkdir(exist_ok=True)
    mask_dst_dir.mkdir(exist_ok=True)
    
    # 2. Read the CSV file
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            filename = row[csv_column]
            
            # --- Move Image ---
            img_src = img_src_dir / filename
            if img_src.exists():
                shutil.move(str(img_src), str(img_dst_dir / filename))
                print(f"Moved image: {filename}")
            else:
                print(f"Image not found: {img_src}")
                
            # --- Move Mask ---
            # Create mask name: e.g., 'pat_01_01.png' -> 'pat_01_01_mask.png'
            mask_name = f"{Path(filename).stem}_mask{Path(filename).suffix}"
            mask_src = mask_src_dir / mask_name
            
            if mask_src.exists():
                shutil.move(str(mask_src), str(mask_dst_dir / mask_name))
                print(f"Moved mask: {mask_name}")
            else:
                print(f"Mask not found: {mask_src}")

if __name__ == "__main__":
    move_files()