import requests
import numpy as np
import cv2


class ImageExtractor:
    def __init__(self, url: str) -> None:
        self.url = url

    def get_image(self) -> np.array:
        image = None
        while image is None:
            try:
                image = self._extract_and_convert_image()
            except Exception as exc:
                print(f"Empty image \n {exc}")
                image = None
        return image

    def _extract_and_convert_image(self) -> np.array:
        response = requests.get(self.url, stream=True)
        while response.status_code != 200:
            response = requests.get(self.url, stream=True)
            print(response.status_code)
        img_array = np.frombuffer(response.content, np.uint8)
        image = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return image
