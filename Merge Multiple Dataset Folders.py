import os
import shutil
from pathlib import Path
from tkinter import filedialog, Tk

def merge_datasets():
    root = Tk()
    root.withdraw()

    print("Select MULTIPLE dataset folders (each must have images & labels)")
    source_dirs = filedialog.askdirectory(title="Select Parent Folder Containing All Datasets")

    output_dir = filedialog.askdirectory(title="Select Output Folder")

    output_images = Path(output_dir) / "images"
    output_labels = Path(output_dir) / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    counter = 0

    for subfolder in Path(source_dirs).iterdir():
        if not subfolder.is_dir():
            continue

        img_dir = subfolder / "images"
        lbl_dir = subfolder / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Skipping {subfolder.name} (invalid structure)")
            continue

        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            label_path = lbl_dir / (img_path.stem + ".txt")

            if not label_path.exists():
                continue

            new_name = f"img_{counter}"

            shutil.copy(img_path, output_images / f"{new_name}{img_path.suffix}")
            shutil.copy(label_path, output_labels / f"{new_name}.txt")

            counter += 1

    print(f"\n✅ Done! Total files merged: {counter}")

if __name__ == "__main__":
    merge_datasets()