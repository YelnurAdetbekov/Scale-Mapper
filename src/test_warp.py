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
for string in compute_frets(choose_scale(['D', "major"])):
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

	if corners is None or len(corners) != 4:
		cv2.imshow("Original", frame)
		continue

	corners = np.asarray(corners, dtype=np.float32)
		
	M, M_inv = build_warp(corners)

	pts = corners.astype(np.int32)

	overlay_lines = np.zeros((WARPED_HEIGHT, WARPED_WIDTH, 3), dtype=np.uint8)
	overlay_rects = np.zeros_like(overlay_lines)
	overlay_centers = np.zeros_like(overlay_lines)

	for fret_x in frets_dict.values():
		cv2.line(overlay_lines, (int(fret_x), WARPED_HEIGHT), (int(fret_x), 0), color=(0,0,255), thickness=2)
	
	for string in fret_boxes:
		for box in string:
			cv2.line(overlay_lines, (0, box[0][1]), (WARPED_WIDTH, box[0][1]), color=(0,0,255), thickness=2)
			cv2.rectangle(overlay_rects, box[0], box[1], color = (0,255,0), thickness=-1)
			
			center_x = (box[0][0] + box[1][0]) // 2
			center_y = (box[0][1] + box[1][1]) // 2
			cv2.circle(overlay_centers, (center_x, center_y), 10, (255,255,0), thickness=-1)

	h, w = frame.shape[:2]
	
	lines_on_frame = cv2.warpPerspective(overlay_lines, M_inv, (w, h))
	rects_on_frame = cv2.warpPerspective(overlay_rects, M_inv, (w, h))
	centers_on_frame = cv2.warpPerspective(overlay_centers, M_inv, (w, h))
	
	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.fillPoly(mask, [pts], 255)
	
	lines_on_frame = cv2.bitwise_and(lines_on_frame, lines_on_frame, mask=mask)
	rects_on_frame = cv2.bitwise_and(rects_on_frame, rects_on_frame, mask=mask)
	centers_on_frame = cv2.bitwise_and(centers_on_frame, centers_on_frame, mask=mask)

	frame = cv2.addWeighted(frame, 1.0, rects_on_frame, 0.3, 0)
	frame = cv2.addWeighted(frame, 1.0, lines_on_frame, 1.0, 0)
	frame = cv2.addWeighted(frame, 1.0, centers_on_frame, 1.0, 0)

	cv2.imshow("Original", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break