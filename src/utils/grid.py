import cv2
import numpy as np

def preprocess(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
	return edges

def find_lines(edges):
	lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=np.pi / 180,
		threshold=85,
		minLineLength=100,
		maxLineGap=20
	)
	return lines if lines is not None else []
def draw_lines(frame, lines):
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
	return frame
def classify_lines(lines):
	horizontal, vertical = [], []
	for line in lines:
		x1, y1, x2, y2 = line[0]
		angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
		if angle < 20:          # nearly horizontal → strings
			horizontal.append(line)
		elif angle > 70:        # nearly vertical → frets
			vertical.append(line)
	return horizontal, vertical