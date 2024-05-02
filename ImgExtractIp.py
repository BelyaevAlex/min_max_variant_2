import numpy as np
import cv2
import requests
from skimage.metrics import structural_similarity


class ImageCapture:
    def __init__(self, camera_ip, logg, **kwargs):
        self._camera_ip = camera_ip
        self._username = kwargs.get('username')
        self._password = kwargs.get('password')
        self._logger = logg
        self._prev_img = None
        if not self._username or not self._password:
            self._logger.warning("Empty password or username")

    def get_snapshot(self):
        try:
            resp = requests.get(self._camera_ip)
            img_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            ssim_value = 0.
            if self._prev_img is not None:
                self._prev_img = cv2.cvtColor(self._prev_img, cv2.COLOR_BGR2GRAY)
                image_cp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                ssim_value = structural_similarity(self._prev_img, image_cp, full=True)[0]
            self._prev_img = image
            return image, ssim_value
        except Exception as exc:
            self._logger.warning(f"Empty image.\n {exc} \n Skipping iteration...")
            return None, None