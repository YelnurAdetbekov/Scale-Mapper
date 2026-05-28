from ultralytics import YOLO
import numpy as np
from collections import deque
from warp import extract_corners
import cv2

SMALL_X, SMALL_Y = 640, 360

class FretboardDetector:
	def __init__(self, model_path):
		self.model = YOLO(model_path)
		self.history = deque(maxlen=2)
		self.history.append(np.zeros((4,2), dtype=np.float32))
		self.history.append(np.zeros((4,2), dtype=np.float32))

	def detect(self, frame, conf=0.5):
		mask_coordinates = self.model(frame, verbose = False, device = "mps")[0]

		if(mask_coordinates.masks is None):
			return np.zeros((4,2), dtype=np.float32)

		idx = int(mask_coordinates.boxes.conf.argmax())
		conf_val = float(mask_coordinates.boxes.conf[idx])

		if conf_val<conf:
			return self._smoothed()

		polygon = mask_coordinates.masks.xy[idx] # (N,2)

		corners = extract_corners(polygon) #(4,2)
		self.history.append(corners)
		return self._smoothed()

	def _smoothed(self):
		if not self.history:
			return None
		return (self.history[0]*0.2 + self.history[1]*0.8)