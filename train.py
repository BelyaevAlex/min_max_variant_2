from roboflow import Roboflow
from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8x.pt')

results = model.train(
    data='datasets/data/Boxes updated.v25i.yolov8/data.yaml',
    imgsz=640,
    epochs=100,
    batch=2,
    name='yolov8s_custom',
)
