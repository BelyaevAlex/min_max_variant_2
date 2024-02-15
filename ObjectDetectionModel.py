from ultralytics import YOLO
import torch
import numpy as np


class YOLOv8ObjDetectionModel:
    def __init__(self, model_path: str) -> None:
        self._model = YOLO(model_path)

    @torch.no_grad()
    def __call__(self, img: np.array, conf=0.25) -> list:
        results = self._model.predict(
            source=img,
            conf=conf
        )[0]
        return results
