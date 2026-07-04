import cv2
import numpy as np
from pathlib import Path

def extract_corners(polygon: np.ndarray):

	#top-left = small x, small y
   #bottom-right = large x, large y
	pts = polygon.astype(np.float32) #OpenCV perspective functions require float32.
	x, y = pts[:,0], pts[:,1]
	# print(x, "\n", y)
	tl = pts[np.argmin(x + y)]   # min(x+y)
	tr  = pts[np.argmax(x - y)]   # max(x-y)
	br  = pts[np.argmax(x + y)]   # max(x+y)
	bl  = pts[np.argmin(x - y)]   # min(x-y)
	return np.array([tl, tr, br, bl], dtype = np.float32)

def build_warp(corners: np.ndarray, out_w=1280, out_h=166):
	dst = np.array([ # A straight rectangle
		[0, 0], 
		[out_w, 0],
		[out_w, out_h],
		[0, out_h]
	], dtype=np.float32) 
	return (cv2.getPerspectiveTransform(corners, dst), 
			cv2.getPerspectiveTransform(dst, corners))

def warp_frame(frame, M, out_w=1280, out_h=166):
	return cv2.warpPerspective(frame, M, (out_w, out_h)) #src, M, size

def project_point_black(pt, M_inv):
	p = np.array([[[pt[0], pt[1]]]], dtype=np.float32) #specific point to map
	back = cv2.perspectiveTransform(p, M_inv) #warped coords -> original coords
	return tuple(back[0][0].astype(int))


def load_camera_calibration(calibration_path):
	calibration_path = Path(calibration_path)
	if not calibration_path.exists():
		return None, None

	data = np.load(calibration_path)
	return data["camera_matrix"], data["dist_coeffs"]


def undistort_frame(frame, camera_matrix=None, dist_coeffs=None):
	if camera_matrix is None or dist_coeffs is None:
		return frame

	return cv2.undistort(frame, camera_matrix, dist_coeffs)