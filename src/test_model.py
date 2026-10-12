from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

MODEL = "runs/segment/models/runs/fretboard_seg_v2/weights/best.pt"
model = YOLO(MODEL)
path = Path("data/raw_images/img_0005.jpg")
img = cv2.imread(path)

results = model(img, verbose = False)

polygons = [mask.astype(np.int32) for mask in results[0].masks.xy]
cv2.polylines(img, polygons, isClosed=True,
								color=(0,255,0), thickness=2)
cv2.imshow("check", img)
cv2.waitKey(0)