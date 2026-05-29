import cv2
from src.camera import get_camera

def collect_dataset(save_dir = "data/raw_images", target = 30):
	import time
	import os
	os.makedirs(save_dir, exist_ok=True)
	cap = get_camera()
	count = 0
	print(f"Collecting {target} images. Press SPACE to capture, Q to quit.")

	while count < target:
		ret, frame = cap.read()
		if not ret:
			break
		frame = cv2.flip(frame, 1)
		display = frame.copy()
		cv2.putText(display, f"{count}/{target} — SPACE to capture",
						(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
		cv2.imshow("Collect", display)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(' '):
			path = f"{save_dir}/img_{count:04d}.jpg"
			cv2.imwrite(path, frame)
			count += 1
			print(f"Saved {path}")
			time.sleep(0.1)
		elif key == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	collect_dataset()