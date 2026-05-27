import cv2, os, random
import numpy as np

img_dir = "data/dataset/train/images"
lbl_dir = "data/dataset/train/labels"

for fname in random.sample(os.listdir(img_dir), 8):
	img  = cv2.imread(f"{img_dir}/{fname}")
	h, w = img.shape[:2]
	lbl  = f"{lbl_dir}/{fname.replace('.jpg','.txt')}"

	with open(lbl) as f:
		for line in f:
			vals = list(map(float, line.split()))
			# segmentation: class_id x1 y1 x2 y2 x3 y3 ...
			coords = np.array(vals[1:]).reshape(-1, 2)
			coords[:, 0] *= w
			coords[:, 1] *= h
			pts = coords.astype(np.int32)
			cv2.polylines(img, [pts], isClosed=True,
								color=(0,255,0), thickness=2)

	cv2.imshow(f"Label check {lbl}", img)
	if cv2.waitKey(0) & 0xFF == ord('q'): break

cv2.destroyAllWindows()