# organize_dataset.py
import os
import shutil
from pathlib import Path
import random

# Tell the script where your files are currently located
source_folder = "sahilnishad@SAHILs-MacBook-Air dataset"  # CHANGE THIS to your folder path

# Where to create the organized dataset
destination = "organized_dataset"

def organize_files(source, dest):
    # Create the necessary folders
    os.makedirs(f"{dest}/images/train", exist_ok=True)
    os.makedirs(f"{dest}/images/val", exist_ok=True)
    os.makedirs(f"{dest}/labels/train", exist_ok=True)
    os.makedirs(f"{dest}/labels/val", exist_ok=True)
    
    # Get all image files
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(Path(source).glob(f'**/{ext}'))
        images.extend(Path(source).glob(f'**/{ext.upper()}'))
    
    print(f"Found {len(images)} images")
    
    # Separate images that have matching .txt files
    valid_pairs = []
    for img in images:
        txt_file = img.with_suffix('.txt')
        if txt_file.exists():
            valid_pairs.append((img, txt_file))
    
    print(f"Found {len(valid_pairs)} images with matching label files")
    
    # Shuffle and split (80% train, 20% val)
    random.shuffle(valid_pairs)
    split_point = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:split_point]
    val_pairs = valid_pairs[split_point:]
    
    # Copy files to train folder
    print(f"\nCopying {len(train_pairs)} images to train set...")
    for img_path, txt_path in train_pairs:
        # Copy image
        shutil.copy(img_path, f"{dest}/images/train/{img_path.name}")
        # Copy label
        shutil.copy(txt_path, f"{dest}/labels/train/{txt_path.name}")
        print(f"  ✓ {img_path.name}")
    
    # Copy files to validation folder
    print(f"\nCopying {len(val_pairs)} images to validation set...")
    for img_path, txt_path in val_pairs:
        # Copy image
        shutil.copy(img_path, f"{dest}/images/val/{img_path.name}")
        # Copy label
        shutil.copy(txt_path, f"{dest}/labels/val/{txt_path.name}")
        print(f"  ✓ {img_path.name}")
    
    print(f"\n✅ Done! Dataset created in '{dest}' folder")
    print(f"   Train: {len(train_pairs)} images")
    print(f"   Validation: {len(val_pairs)} images")

# Run the organization
organize_files(source_folder, destination)