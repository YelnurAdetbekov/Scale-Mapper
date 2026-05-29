import cv2
from camera import get_camera
from detect import FretboardDetector
from warp import build_warp, warp_frame
import numpy as np
from grid import compute_stretched_fretboard, compute_frets, choose_scale

MODEL = "runs/segment/models/runs/fretboard_seg_v2/weights/best.pt"
WARPED_HEIGHT = 166
WARPED_WIDTH = 1280

detector = FretboardDetector(MODEL)
cap = get_camera()
frets_dict = compute_stretched_fretboard()

string_offset = 0
temp = 0
fret_boxes = [[],[],[],[],[],[]]
i = 0
for string in compute_frets(choose_scale(['A', "minor"])):
	temp = string_offset
	string_offset += WARPED_HEIGHT/6
	for fret in string:
		points = (int(frets_dict[fret-1]), int(temp)), (int(frets_dict[fret]),int(string_offset))
		fret_boxes[i].append(points)
	i += 1
print(fret_boxes)
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
			cv2.line(warped, (int(fret_x), WARPED_HEIGHT), (int(fret_x), 0), color=(0,0,255), thickness = 2)
		overlay = warped.copy()
		for string in fret_boxes:
			for box in string:
				cv2.line(warped, (0, box[0][1]), (1280, box[0][1]), color=(0,0,255), thickness=2)
				cv2.rectangle(overlay, box[0], box[1], color = (0,255,0), thickness=-1)
		alpha = 0.3
		warped = cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0)
		cv2.imshow("Warped", warped)

	cv2.imshow("Original", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break