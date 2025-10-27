#  train_court_yolov8.py
from ultralytics import YOLO
import os
from datetime import datetime

# === Training Parameters Setup ===
model_name = "yolov8x-pose.pt"
data_yaml = "/users/k24016446/basketball_analysis/training_notebook/court_keypoint/reloc2-1/data.yaml"
epochs = 500
imgsz = 640
batch = 4  # Change to 'batch'
workers = 8

# === Training Execution ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"court_train_{timestamp}"
model = YOLO(model_name)

print(f"\n🚀 Starting YOLOv8 Training: {run_name}\n")
model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    name=run_name,
    batch=batch,  # Correct 'batch' instead of 'batch_size'
    workers=workers
)

# === Backup best.pt ===
best_pt = f"runs/pose/{run_name}/weights/best.pt"
os.makedirs("model_checkpoints", exist_ok=True)
backup_path = f"model_checkpoints/best_court_{timestamp}.pt"
if os.path.exists(best_pt):
    os.system(f"cp {best_pt} {backup_path}")
    print(f"\n✅ Best model saved to: {backup_path}\n")
else:
    print("⚠️ Best.pt not found, please check if training was successful")