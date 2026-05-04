# detect_classes.py
import os

def detect_classes_from_labels(labels_path):
    classes = set()
    
    # Look through all .txt files in train labels
    for filename in os.listdir(os.path.join(labels_path, 'train')):
        if filename.endswith('.txt'):
            with open(os.path.join(labels_path, 'train', filename), 'r') as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        class_id = int(line.strip().split()[0])
                        classes.add(class_id)
    
    # Get sorted list of class IDs and create names
    class_ids = sorted(list(classes))
    print(f"Found {len(class_ids)} classes with IDs: {class_ids}")
    
    # Create generic class names (you should rename these appropriately)
    class_names = [f"class_{i}" for i in class_ids]
    print(f"Class names: {class_names}")
    
    return len(class_ids), class_names

# Run the detection
labels_path = './dataset/labels'
num_classes, class_names = detect_classes_from_labels(labels_path)

# Update your data.yaml with these values
print(f"\nUpdate your data.yaml with:")
print(f"nc: {num_classes}")
print(f"names: {class_names}")