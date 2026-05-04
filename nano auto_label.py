from ultralytics import YOLO
import os
from pathlib import Path
import cv2  # ✅ ADD THIS

# Load your trained model
model = YOLO('C:/Users/sahil.k_droisys/Desktop/tool2/object_detection/custom_training/weights/best.pt')

input_dir = 'C:/Users/sahil.k_droisys/Desktop/tool2/new_images_to_label'
output_dir = 'C:/Users/sahil.k_droisys/Desktop/tool2/auto_generated_labels'
viz_dir = 'C:/Users/sahil.k_droisys/Desktop/tool2/auto_labeled_visualizations'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

if not os.path.exists(input_dir):
    print(f"❌ ERROR: Input directory '{input_dir}' does not exist!")
    exit(1)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = [f for f in os.listdir(input_dir) 
               if any(f.lower().endswith(ext) for ext in image_extensions)]

print(f"📸 Found {len(image_files)} images to process")
print("-" * 50)

total_images = len(image_files)
images_with_detections = 0
total_objects = 0

for image_file in image_files:
    print(f"Processing: {image_file}")
    
    image_path = os.path.join(input_dir, image_file)
    results = model(image_path, conf=0.25)
    result = results[0]
    
    if len(result.boxes) > 0:
        images_with_detections += 1
        num_objects = len(result.boxes)
        total_objects += num_objects
        
        # ✅ SAVE LABELS
        label_filename = Path(image_file).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for box in result.boxes:
                class_id = int(box.cls[0])
                xywhn = box.xywhn[0].tolist()
                
                line = f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                f.write(line)
        
        print(f"  ✓ Saved labels: {label_filename}")
        
        # ✅ FIX: SAVE IMAGE (REPLACEMENT FOR result.save())
        img = result.plot()  # draw boxes
        save_path = os.path.join(viz_dir, f"labeled_{image_file}")
        cv2.imwrite(save_path, img)
        
        print(f"  ✓ Saved visualization: labeled_{image_file}")
        
    else:
        print(f"  ✗ No objects detected")
    
    print("-" * 30)

# SUMMARY
print("\n" + "=" * 50)
print("📊 AUTO-LABELING COMPLETE!")
print("=" * 50)
print(f"Total images: {total_images}")
print(f"With detections: {images_with_detections}")
print(f"Without detections: {total_images - images_with_detections}")
print(f"Total objects: {total_objects}")
print("=" * 50)