from ultralytics import YOLO
import numpy as np
from collections import deque
from warp import extract_corners
import cv2
import time

# SMALL_X, SMALL_Y = 640, 360

class FretboardDetector:
	def __init__(self, model_path):
		self.model = YOLO(model_path)
		self.history = deque(maxlen=3)

		self.prev_gray = None
		self.prev_points = None
		self.corners = None
		self.last_yolo_time = 0.0

	# def _initialize_tracker(self, gray):
		
	def _track(self, gray):

		next_points, status, err = cv2.calcOpticalFlowPyrLK(
			self.prev_gray,
			gray,
			self.prev_points,
			None
		)
		
		if next_points is None:
			self._reset()
			return False, self._smoothed()
		
		status = status.reshape(-1)

		good_new = next_points[status == 1]
		good_old = self.prev_points[status == 1]

		if len(good_new) < 3:
			self._reset()
			return False, self._smoothed()
		
		motion = np.linalg.norm(good_new - good_old, axis=1).mean()
		if motion > 30:  # tune this
			self._reset()
			return False, None
		
		M, inliers = cv2.estimateAffinePartial2D(
			good_old,
			good_new
		)

		if M is None:
			self._reset()
			return False, None
		
		if abs(M[0,2]) > 100 or abs(M[1,2]) > 100:
			self._reset()
			return False, None
		
		self.corners = cv2.transform(
			self.corners.reshape(-1,1,2),
			M
		).reshape(-1,2)

		self.prev_gray = gray
		self.prev_points = good_new.reshape(-1,1,2)

		self.history.append(self.corners)
		return True, self._smoothed()

	def _reset(self):
		self.prev_gray = None
		self.prev_points = None
		self.corners = None

	def detect(self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		now = time.monotonic()
		
		if self.prev_gray is None or (now - self.last_yolo_time) >= 4.0:
			return self._detect_with_yolo(frame, gray)
		
		ok, corners = self._track(gray)

		if not ok:
			return self._detect_with_yolo(frame, gray)
		
		return corners
	
	def _detect_with_yolo(self, frame, gray, conf=0.5):
		self.last_yolo_time = time.monotonic()

		# h, w = frame.shape[:2]
		# small = cv2.resize(frame, (SMALL_X, SMALL_Y))

		mask_coordinates = self.model(frame, verbose = False, device = "mps")[0]

		if(mask_coordinates.masks is None or len(mask_coordinates) == 0):
			return self._smoothed()

		idx = int(mask_coordinates.boxes.conf.argmax())
		conf_val = float(mask_coordinates.boxes.conf[idx])

		if conf_val<conf:
			return self._smoothed()

		polygon = mask_coordinates.masks.xy[idx].copy() # (N,2)
		# polygon[:, 0] *= w / SMALL_X
		# polygon[:, 1] *= h / SMALL_Y

		self.corners = extract_corners(polygon) #(4,2)

		mask = np.zeros(gray.shape, dtype=np.uint8)
		cv2.fillConvexPoly(mask,
								self.corners.astype(np.int32),
								255)

		points = cv2.goodFeaturesToTrack(
			gray,
			maxCorners=200,
			qualityLevel=0.01,
			minDistance=8,
			mask=mask
		)

		if points is None:
			return self._smoothed()

		self.prev_gray = gray.copy()
		self.prev_points = points

		self.history.append(self.corners)
		return self._smoothed()

	def _smoothed(self):
		if not self.history:
			return None
		return np.mean(self.history, axis=0).astype(np.float32)