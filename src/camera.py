import cv2
import utils.grid as gr

def get_camera(index = 1):
	cap = cv2.VideoCapture(index)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 30)	
	return cap

def run_preview():
	cap = get_camera()
	print("Press Q to quit, S to save a frame")
	frame_count = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame = cv2.flip(frame, 1)
		edges = gr.preprocess(frame)
		lines = gr.find_lines(edges)
		hor, ver = gr.classify_lines(lines)
		#frame = gr.draw_lines(frame, hor)
		frame = gr.draw_lines(frame, ver)
		cv2.imshow("Camera preview", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('s'):
			path = f"data/raw_images/frame_{frame_count:04d}.jpg"
			cv2.imwrite(path, frame)
			print(f"Saved {path}")
		frame_count += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	run_preview()