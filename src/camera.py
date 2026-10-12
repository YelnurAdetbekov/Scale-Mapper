import cv2
import platform
from cv2_enumerate_cameras import enumerate_cameras

def get_camera():
	camera = get_default_camera()
	cap = cv2.VideoCapture(index)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 40)	
	return cap

