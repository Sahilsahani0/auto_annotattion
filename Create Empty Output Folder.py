import os
import shutil
from pathlib import Path

# 👉 CHANGE THIS PATH (your main folder)
SOURCE = r"//LV-NAS-Droisys/account360_ai_image_backup/Monu Backup/DataSet/Sazerac_Dataset/Labelled"

# 👉 CHANGE OUTPUT PATH
OUTPUT = r"//LV-NAS-Droisys/account360_ai_image_backup/Monu Backup/DataSet/Sazerac_Dataset/final_dataset"

images_out = Path(OUTPUT) / "images"
labels_out = Path(OUTPUT) / "labels"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

class_id = 0
counter = 0

for folder in os.listdir(SOURCE):
    folder_path = Path(SOURCE) / folder

    if not folder_path.is_dir():
        continue

    print(f"Processing: {folder} → class {class_id}")

    for file in folder_path.iterdir():

        if file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        label_file = folder_path / (file.stem + ".txt")

        if not label_file.exists():
            continue

        new_name = f"img_{counter}"

        # copy image
        shutil.copy(file, images_out / f"{new_name}{file.suffix}")

        # update label
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = str(class_id)
                new_lines.append(" ".join(parts))

        with open(labels_out / f"{new_name}.txt", "w") as f:
            f.write("\n".join(new_lines))

        counter += 1

    class_id += 1

print("\nDONE ✅")