from ultralytics import YOLO
import torch
import numpy as np


class YOLOv8ObjDetectionModel:
    def __init__(self, model_path: str, classes: list = None) -> None:
        self._model = YOLO(model_path)
        self._classes = classes

    @torch.no_grad()
    def __call__(self, img: np.array, conf=0.5):
        if self._classes:
            results = self._model.predict(
                source=img,
                conf=conf,
                classes=self._classes
            )[0]
            return results
        else:
            results = self._model.predict(
                source=img,
                conf=conf
            )[0]
            return results
