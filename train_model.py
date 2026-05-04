# train_model.py
from ultralytics import YOLO
import torch

# Check if MPS (Apple GPU) is available
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Load a pretrained model (using nano version for Mac efficiency)
model = YOLO('yolov8n.pt')  # you can also use 'yolov8s.pt' for better accuracy

# Train the model
results = model.train(
    data='C:/Users/sahil.k_droisys/Desktop/tool2/nano data.yaml',           # path to your data.yaml
    epochs=20,                  # number of training epochs
    imgsz=640,                   # input image size
    batch=8,                     # batch size (reduce to 4 if you have memory issues)
    device='mps',                # use Apple Silicon GPU
    workers=4,                   # number of worker threads
    patience=20,                  # early stopping patience
    save=True,                   # save checkpoints
    project='object_detection',   # project name
    name='custom_training',       # experiment name
    exist_ok=True,               # overwrite existing experiment
    pretrained=True,             # use pretrained weights
    optimizer='auto',            # automatically choose optimizer
    verbose=True                 # print detailed output
)

print("Training complete! Model saved in 'object_detection/custom_training/weights/best.pt'")