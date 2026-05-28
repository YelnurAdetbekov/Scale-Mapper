from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model.train(
	data = "data/dataset/data.yaml",
	epochs = 60,
	imgsz = 640,
	batch = 8,
	device = "mps",
	project = "models/runs",
	name = "fretboard_seg_v1",
	patience = 15,
	cache = True,
	augment = True
)