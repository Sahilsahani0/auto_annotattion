import os

label_dirs = [
    r"C:\Users\sahil.k_droisys\Desktop\tool2\my_object_detector\dataset\labels\train",
    r"C:\Users\sahil.k_droisys\Desktop\tool2\my_object_detector\dataset\labels\val"
]

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            path = os.path.join(label_dir, file)

            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # force class = 0
                    new_lines.append(" ".join(parts))

            with open(path, "w") as f:
                f.write("\n".join(new_lines))

print("✅ All labels fixed to class 0")