import cv2

def get_camera(index = 1):
	cap = cv2.VideoCapture(index)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 40)	
	return cap

