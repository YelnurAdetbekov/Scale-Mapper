from ultralytics import YOLO
import numpy as np
from warp import extract_corners
import cv2
import time
from sklearn.linear_model import LinearRegression, RANSACRegressor

# SMALL_X, SMALL_Y = 640, 360
FEATURE_THRESHOLD = 80

class FretboardDetector:
	def __init__(self, model_path):
		self.model = YOLO(model_path)
		self.smoothing_alpha = 0.75

		self.prev_gray = None
		self.prev_points = None
		self.corners = None
		self.last_yolo_time = 0.0
		
	def _track(self, gray):

		next_points, status, err = cv2.calcOpticalFlowPyrLK(
			self.prev_gray,
			gray,
			self.prev_points,
			None,
			maxLevel=4
		)
		
		if next_points is None:
			self._reset()
			return False, self._smoothed()
		
		status = status.reshape(-1)

		good_new = next_points[(status == 1) & (err.flatten() < 10)]
		good_old = self.prev_points[(status == 1) & (err.flatten() < 10)]

		if len(good_new) < 30:
			self._reset()
			return False, self._smoothed()
		
		M, inliers = cv2.findHomography(
			good_old,
			good_new,
			cv2.RANSAC,
			3.0
		)

		if M is None:
			self._reset()
			return False, None
		
		inliers_mask = inliers.ravel().astype(bool)
		good_old = good_old[inliers_mask]
		good_new = good_new[inliers_mask]

		if abs(M[0,2]) > 100 or abs(M[1,2]) > 100:
			self._reset()
			return False, None
		
		new_corners = cv2.perspectiveTransform(
			self.corners.reshape(-1,1,2),
			M
		).reshape(-1,2)

		if not self._is_reasonable_corners(new_corners, reference=self.corners, image_shape=gray.shape):
			self._reset()
			return False, None

		self.corners = new_corners

		self.prev_gray = gray
		if len(good_new) < FEATURE_THRESHOLD:
			replenished = cv2.goodFeaturesToTrack(
				gray,
				maxCorners=200,
				qualityLevel=0.01,
				minDistance=8,
				mask=self._feature_mask_from_corners(gray.shape),
			)
			if replenished is not None:
				good_new = np.vstack((good_new.reshape(-1, 1, 2), replenished.reshape(-1, 1, 2)))
			else:
				good_new = good_new.reshape(-1, 1, 2)
		else:
			good_new = good_new.reshape(-1, 1, 2)
		self.prev_points = good_new.reshape(-1,1,2)

		return True, self._smoothed()

	def _reset(self):
		self.prev_gray = None
		self.prev_points = None
		self.corners = None

	def detect(self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		now = time.monotonic()
		
		if self.prev_gray is None or (now - self.last_yolo_time) >= 4:
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

		# print(mask_coordinates.masks)
		idx = int(mask_coordinates.boxes.conf.argmax())
		conf_val = float(mask_coordinates.boxes.conf[idx])

		if conf_val<conf:
			return self._smoothed()

		polygon = mask_coordinates.masks.xy[idx].copy() # (N,2)
		# polygon[:, 0] *= w / SMALL_X
		# polygon[:, 1] *= h / SMALL_Y

		corners = extract_corners(polygon) #(4,2)
		refined_corners = self._refine_corners_with_lines(polygon, corners, gray)
		print(refined_corners, "\n")
		if refined_corners is not None and self._is_reasonable_corners(refined_corners, reference=corners, image_shape=gray.shape):
			if self.corners is None:
				self.corners = refined_corners
			else:
				self.corners = 0.6 * self.corners + 0.4 * refined_corners
		else:
			self.corners = self._smooth_corners(corners)

		feature_mask = None
		if mask_coordinates.masks is not None:
			feature_mask = mask_coordinates.masks.data[idx].detach().cpu().numpy()
			if feature_mask.shape != gray.shape:
				feature_mask = cv2.resize(
					feature_mask.astype(np.uint8),
					(gray.shape[1], gray.shape[0]),
					interpolation=cv2.INTER_NEAREST,
				)
			feature_mask = (feature_mask > 0).astype(np.uint8) * 255

		mask = np.zeros(gray.shape, dtype=np.uint8)
		cv2.fillConvexPoly(
			mask,
			self.corners.astype(np.int32),
			255,
		)
		if feature_mask is not None:
			mask = cv2.bitwise_and(mask, feature_mask)
			if not np.any(mask):
				mask = feature_mask

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

		return self._smoothed()

	def _refine_corners_with_lines(self, polygon, corners, gray):
		height, width = gray.shape[:2]
		mask = np.zeros((height, width), dtype=np.uint8)
		cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			return None

		contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
		if len(contour) < 80:
			return None

		center = contour.mean(axis=0)
		offset_points = contour - center
		cov = np.cov(offset_points.T)
		eigenvalues, eigenvectors = np.linalg.eigh(cov)
		long_axis = eigenvectors[:, np.argmax(eigenvalues)].astype(np.float32)
		short_axis = eigenvectors[:, np.argmin(eigenvalues)].astype(np.float32)

		long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-6)
		short_axis = short_axis / max(float(np.linalg.norm(short_axis)), 1e-6)

		u = offset_points @ long_axis
		v = offset_points @ short_axis

		line_specs = [
			(v, u, short_axis, long_axis, np.percentile(v, 20.0), "min_v"),
			(v, u, short_axis, long_axis, np.percentile(v, 80.0), "max_v"),
			(u, v, long_axis, short_axis, np.percentile(u, 20.0), "min_u"),
			(u, v, long_axis, short_axis, np.percentile(u, 80.0), "max_u"),
		]

		lines = {}
		for primary, secondary, primary_axis, secondary_axis, threshold, key in line_specs:
			if key.startswith("min_"):
				selected = contour[(primary <= threshold + 0.02 * (primary.max() - primary.min()))]
			else:
				selected = contour[(primary >= threshold - 0.02 * (primary.max() - primary.min()))]

			line = self._fit_ransac_line(selected, center, primary_axis, secondary_axis)
			if line is None:
				return None
			lines[key] = line

		corners_from_lines = [
			self._line_intersection(lines["min_v"][0], lines["min_v"][1], lines["min_u"][0], lines["min_u"][1]),
			self._line_intersection(lines["min_v"][0], lines["min_v"][1], lines["max_u"][0], lines["max_u"][1]),
			self._line_intersection(lines["max_v"][0], lines["max_v"][1], lines["max_u"][0], lines["max_u"][1]),
			self._line_intersection(lines["max_v"][0], lines["max_v"][1], lines["min_u"][0], lines["min_u"][1]),
		]
		if any(corner is None for corner in corners_from_lines):
			return None

		refined = extract_corners(np.asarray(corners_from_lines, dtype=np.float32))
		if cv2.contourArea(refined.astype(np.float32)) <= 0:
			return None

		return refined

	def _fit_ransac_line(self, points, center, direction_axis, normal_axis):
		if len(points) < 30:
			return None

		direction_axis = np.asarray(direction_axis, dtype=np.float32)
		normal_axis = np.asarray(normal_axis, dtype=np.float32)
		relative = points - center
		x = relative @ direction_axis
		y = relative @ normal_axis

		if len(np.unique(np.round(x, 6))) < 2:
			return None

		model = RANSACRegressor(
			estimator=LinearRegression(),
			min_samples=max(10, len(x) // 3),
			residual_threshold=2.5,
			random_state=0,
		)
		try:
			model.fit(x.reshape(-1, 1), y)
		except ValueError:
			return None

		inlier_mask = model.inlier_mask_
		if inlier_mask is None or int(np.sum(inlier_mask)) < 20:
			return None

		coef = float(model.estimator_.coef_[0])
		intercept = float(model.estimator_.intercept_)
		point = center + normal_axis * intercept
		direction = direction_axis + normal_axis * coef
		direction_norm = float(np.linalg.norm(direction))
		if direction_norm < 1e-6:
			return None

		return np.array(point, dtype=np.float32), (direction / direction_norm).astype(np.float32)

	def _is_reasonable_corners(self, corners, reference=None, image_shape=None):
		corners = np.asarray(corners, dtype=np.float32)
		if corners.shape != (4, 2) or not np.isfinite(corners).all():
			return False

		area = abs(cv2.contourArea(corners.reshape(-1, 1, 2)))
		if area < 1000:
			return False

		if image_shape is not None:
			height, width = image_shape[:2]
			if np.any(corners[:, 0] < -0.25 * width) or np.any(corners[:, 0] > 1.25 * width):
				return False
			if np.any(corners[:, 1] < -0.25 * height) or np.any(corners[:, 1] > 1.25 * height):
				return False

		if reference is not None:
			reference = np.asarray(reference, dtype=np.float32)
			reference_area = abs(cv2.contourArea(reference.reshape(-1, 1, 2)))
			if reference_area > 0:
				ratio = area / reference_area
				if ratio < 0.35 or ratio > 2.5:
					return False

			centroid = corners.mean(axis=0)
			reference_centroid = reference.mean(axis=0)
			distance = float(np.linalg.norm(centroid - reference_centroid))
			if distance > max(60.0, 0.15 * np.linalg.norm(reference.max(axis=0) - reference.min(axis=0))):
				return False

		return True

	def _line_intersection(self, point_a, direction_a, point_b, direction_b):
		x1, y1 = point_a
		dx1, dy1 = direction_a
		x2, y2 = point_b
		dx2, dy2 = direction_b

		denominator = dx1 * dy2 - dy1 * dx2
		if abs(denominator) < 1e-6:
			return None

		t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denominator
		return np.array([x1 + t * dx1, y1 + t * dy1], dtype=np.float32)

	def _smoothed(self):
		if self.corners is None:
			return None
		return self.corners.astype(np.float32)

	def _feature_mask_from_corners(self, image_shape):
		mask = np.zeros(image_shape, dtype=np.uint8)
		if self.corners is not None:
			cv2.fillConvexPoly(mask, self.corners.astype(np.int32), 255)
		return mask

	def _smooth_corners(self, new_corners):
		new_corners = np.asarray(new_corners, dtype=np.float32)
		if self.corners is None:
			return new_corners

		return (1.0 - self.smoothing_alpha) * self.corners + self.smoothing_alpha * new_corners