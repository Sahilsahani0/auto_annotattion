# advanced_auto_label.py
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

class AutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def process_images(self, input_dir, output_dir, visualize=True):
        """Process all images and auto-generate labels"""
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            os.makedirs(f"{output_dir}_visualized", exist_ok=True)
        
        image_files = self._get_image_files(input_dir)
        stats = {'processed': 0, 'with_detections': 0, 'total_objects': 0}
        
        for image_file in image_files:
            result = self._process_single_image(
                os.path.join(input_dir, image_file), 
                output_dir, 
                visualize
            )
            
            stats['processed'] += 1
            if result['has_detections']:
                stats['with_detections'] += 1
                stats['total_objects'] += result['object_count']
            
            print(f"Processed {stats['processed']}/{len(image_files)}: {image_file} - "
                  f"{result['object_count']} objects")
        
        return stats
    
    def _process_single_image(self, image_path, output_dir, visualize):
        """Process a single image"""
        results = self.model(image_path, conf=self.confidence_threshold)
        result = results[0]
        
        image_file = Path(image_path).name
        stats = {'has_detections': False, 'object_count': 0}
        
        if len(result.boxes) > 0:
            # Filter by confidence if needed
            high_conf_boxes = [box for box in result.boxes 
                             if box.conf[0] >= self.confidence_threshold]
            
            if high_conf_boxes:
                stats['has_detections'] = True
                stats['object_count'] = len(high_conf_boxes)
                
                # Save YOLO format labels
                label_path = os.path.join(output_dir, Path(image_file).stem + '.txt')
                with open(label_path, 'w') as f:
                    for box in high_conf_boxes:
                        class_id = int(box.cls[0])
                        xywhn = box.xywhn[0].tolist()
                        conf = float(box.conf[0])
                        f.write(f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} "
                               f"{xywhn[2]:.6f} {xywhn[3]:.6f}  # conf: {conf:.3f}\n")
                
                # Save visualization
                if visualize:
                    result.save(filename=os.path.join(f"{output_dir}_visualized", 
                                                      f"labeled_{image_file}"))
        
        return stats
    
    def _get_image_files(self, directory):
        """Get all image files from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return [f for f in os.listdir(directory) 
                if any(f.lower().endswith(ext) for ext in extensions)]

# Usage
labeler = AutoLabeler(
    model_path='object_detection/custom_training/weights/best.pt',
    confidence_threshold=0.6  # only keep detections with 60%+ confidence
)

stats = labeler.process_images(
    input_dir='new_images',
    output_dir='auto_labels',
    visualize=True
)

print(f"\n=== Auto-labeling Complete ===")
print(f"Total images processed: {stats['processed']}")
print(f"Images with detections: {stats['with_detections']}")
print(f"Total objects detected: {stats['total_objects']}")
print(f"Average objects per image: {stats['total_objects']/stats['processed']:.2f}")