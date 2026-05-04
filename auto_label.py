# auto_label.py
from ultralytics import YOLO
import os
from pathlib import Path

# Load your trained model
model = YOLO('object_detection/custom_training/weights/best.pt')

# Directory containing new images to auto-label
input_dir = 'new_images_to_label'
# Directory to save auto-generated labels
output_dir = 'auto_generated_labels'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = [f for f in os.listdir(input_dir) 
               if any(f.lower().endswith(ext) for ext in image_extensions)]

print(f"Found {len(image_files)} images to process")

for image_file in image_files:
    # Run inference
    results = model(os.path.join(input_dir, image_file))
    
    # Get the first result (since we're processing one image at a time)
    result = results[0]
    
    # Generate YOLO format labels
    if len(result.boxes) > 0:  # if objects were detected
        # Get the filename without extension for the label file
        label_filename = Path(image_file).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        # Extract and save labels in YOLO format
        with open(label_path, 'w') as f:
            for box in result.boxes:
                # Get class ID and normalized coordinates
                class_id = int(box.cls[0])
                xywhn = box.xywhn[0].tolist()  # normalized [x_center, y_center, width, height]
                
                # Write in YOLO format: class_id x_center y_center width height
                line = f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                f.write(line)
        
        print(f"✓ Generated labels for {image_file} - found {len(result.boxes)} objects")
        
        # Optional: Save visualization image with bounding boxes
        result.save(filename=os.path.join('auto_labeled_visualizations', f"labeled_{image_file}"))
    else:
        print(f"✗ No objects detected in {image_file}")

print(f"\nAuto-labeling complete! Labels saved in '{output_dir}'")