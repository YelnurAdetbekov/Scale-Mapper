import cv2
from camera import get_camera
from detect import FretboardDetector
from warp import build_warp, warp_frame
import numpy as np
from grid import compute_stretched_fretboard

MODEL = "runs/segment/models/runs/fretboard_seg_v2/weights/best.pt"
WARPED_HEIGHT = 163
WARPED_WIDTH = 1280

detector = FretboardDetector(MODEL)
cap = get_camera()
frets_dict = compute_stretched_fretboard()

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
		for fret_x in frets_dict.values():
			cv2.line(warped, (int(fret_x), WARPED_HEIGHT), (int(fret_x), 0), color=(0,0,255))
		cv2.imshow("Warped", warped)
	
	cv2.imshow("Original", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break