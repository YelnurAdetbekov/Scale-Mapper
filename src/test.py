import os, random
import cv2

img_dir = "data/dataset/train/images"
lbl_dir = "data/dataset/train/labels"

samples = random.sample(os.listdir(img_dir), 5)
for fname in samples:
	img = cv2.imread(f"{img_dir}/{fname}")
	h, w = img.shape[:2]
	lbl_path = f"{lbl_dir}/{fname.replace('.jpg', '.txt')}"
	with open(lbl_path) as f:
		for line in f:
			cls, cx, cy, bw, bh = map(float, line.split())
			x1 = int((cx - bw/2) * w)
			y1 = int((cy - bh/2) * h)
			x2 = int((cx + bw/2) * w)
			y2 = int((cy + bh/2) * h)
			cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
	cv2.imshow("check", img)
	cv2.waitKey(0)