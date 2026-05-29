from ultralytics import YOLO
import numpy as np
from collections import deque
from warp import extract_corners
import cv2

SMALL_X, SMALL_Y = 640, 360

class FretboardDetector:
	def __init__(self, model_path):
		self.model = YOLO(model_path)
		self.history = deque(maxlen=7)

	def detect(self, frame, conf=0.5):
		h, w = frame.shape[:2]
		small = cv2.resize(frame, (SMALL_X, SMALL_Y))
		mask_coordinates = self.model(small, verbose = False, device = "mps")[0]

		if(mask_coordinates.masks is None):
			return self._smoothed()

		idx = int(mask_coordinates.boxes.conf.argmax())
		conf_val = float(mask_coordinates.boxes.conf[idx])

		if conf_val<conf:
			return self._smoothed()

		polygon = mask_coordinates.masks.xy[idx].copy() # (N,2)
		polygon[:, 0] *= w / SMALL_X
		polygon[:, 1] *= h / SMALL_Y

		corners = extract_corners(polygon) #(4,2)
		self.history.append(corners)
		return self._smoothed()

	def _smoothed(self):
		if not self.history:
			return None
		return np.mean(self.history, axis=0).astype(np.float32)