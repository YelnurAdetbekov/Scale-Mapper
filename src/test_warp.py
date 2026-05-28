import cv2
from camera import get_camera
from detect import FretboardDetector
from warp import build_warp, warp_frame
import numpy as np

MODEL = "runs/segment/models/runs/fretboard_seg_v1/weights/best.pt"
detector = FretboardDetector(MODEL)
cap = get_camera()

while True:
	ret, frame = cap.read()
	if not ret:
		break
	frame = cv2.flip(frame, 1)
	corners = detector.detect(frame) #nd.array(4,2)
	if corners is not None:
		M, M_inv = build_warp(corners)
		warped = warp_frame(frame, M)

		pts = corners.astype(np.int32)
		cv2.polylines(frame, [pts], True, (0,255,0), 2)
		cv2.imshow("Warped", warped)
	
	cv2.imshow("Original", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break