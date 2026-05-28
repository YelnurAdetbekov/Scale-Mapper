# Guitar Fretboard Scale Mapper — Detailed Build Plan

**Stack:** Python 3.11 · OpenCV · YOLOv8n (Ultralytics) · PyTorch MPS · Roboflow · Tkinter  
**Hardware:** MacBook M2 Air 8GB  
**Goal:** Camera detects your fretboard → ML model gives robust bounding box → classical CV maps the fret grid → scale dots overlaid in real time

---

## Project structure (build this on Day 1)

```
fretboard_mapper/
├── data/
│   ├── raw_images/        # your photos for labeling
│   └── dataset/           # exported from Roboflow (auto-populated)
├── models/
│   └── runs/              # YOLOv8 training outputs land here
├── src/
│   ├── camera.py          # webcam capture loop
│   ├── detect.py          # YOLOv8 inference wrapper
│   ├── grid.py            # fret/string grid math
│   ├── scales.py          # scale/note logic
│   ├── overlay.py         # drawing dots on frame
│   └── ui.py              # Tkinter control panel
├── calibrate.py           # manual corner-click fallback
├── main.py                # entry point
├── requirements.txt
└── notes.md               # your personal lab notebook
```

---

## Week 1 — Environment + camera foundation

### Day 1–2: Python environment

```bash
# Install pyenv to manage Python versions cleanly
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Create project
mkdir fretboard_mapper && cd fretboard_mapper
python -m venv venv
source venv/bin/activate

# Core installs
pip install opencv-python numpy
pip install ultralytics          # YOLOv8 — pulls in PyTorch automatically
pip install roboflow              # for dataset download later
pip install pillow                # image utilities

# Verify MPS (Apple GPU) is available — critical check
python -c "import torch; print(torch.backends.mps.is_available())"
# Must print: True
```

**If MPS prints False:** your PyTorch install is wrong. Run:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Day 3–4: First camera script

Create `src/camera.py`:

```python
import cv2

def get_camera(index=0):
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

        # Show FPS overlay
        cv2.putText(frame, f"Frame {frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
```

Run it. Make sure you see a smooth 30fps feed. If it's laggy, reduce resolution to 640x480 for now.

### Day 5: Learn OpenCV drawing primitives

Write a throwaway script `scratch_draw.py` to get comfortable — you'll use these everywhere:

```python
import cv2
import numpy as np

# Blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cv2.rectangle(canvas, (50, 50), (300, 200), (0, 255, 0), 2)
cv2.circle(canvas, (400, 240), 30, (0, 100, 255), -1)   # filled
cv2.line(canvas, (0, 0), (640, 480), (255, 0, 0), 1)
cv2.putText(canvas, "Hello fretboard", (60, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("draw test", canvas)
cv2.waitKey(0)
```

**Week 1 milestone:** Camera opens, frames display, you can save images, drawing works.

---

## Week 2 — Classical CV experiments (understand the problem before training)

This week you intentionally try to solve fretboard detection WITHOUT ML. This teaches you exactly what the ML model needs to solve later, and gives you a fallback.

### Day 1–2: Edge detection exploration

Create `src/grid.py` (classical approach first):

```python
import cv2
import numpy as np

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to kill noise, keep edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges

def find_lines(edges):
    # rho=1px, theta=1°, threshold=80 votes minimum
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=20
    )
    return lines if lines is not None else []

def draw_lines(frame, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame
```

Run this in a loop on your live camera. Play with the Canny thresholds until you see fret lines clearly. Write what thresholds work for your setup in `notes.md`.

### Day 3: Separate horizontal from vertical lines

Add to `src/grid.py`:

```python
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
```

**What you should see:** ~6 horizontal lines (strings) and ~12–15 vertical lines (frets 0–12 or so). If you see 40+ noisy lines, raise the Hough threshold. If you see 2, lower it.

### Day 4–5: Understand what breaks it

Deliberately break your detection:
- Move the guitar an inch left. Does it still work?
- Change the room light. Does it break?
- Tilt the neck slightly.

Write down every failure mode in `notes.md`. This is your motivation for training a model — the model handles all of these for free.

**Week 2 milestone:** You can visually detect fret/string lines in good conditions. You understand exactly what breaks it.

---

## Week 3 — Data collection and labeling

This week you build the training dataset. Budget: **~3–4 hours of actual work**, spread over the week.

### Day 1: Physical setup for data collection

Mount your camera in a fixed position (phone tripod works, stack of books works). You want:
- Camera 40–70cm above the fretboard
- Angled 30–60° to the neck (not straight down, not horizontal)
- Decent light — a desk lamp aimed at the fretboard from the side is ideal

### Day 2: Collect raw images

Update `src/camera.py` — add a burst capture mode:

```python
def collect_dataset(save_dir="data/raw_images", target=400):
    import os, time
    os.makedirs(save_dir, exist_ok=True)
    cap = get_camera()
    count = 0
    print(f"Collecting {target} images. Press SPACE to capture, Q to quit.")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            break

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
```

**Vary these while shooting:**
- Guitar position (5 different spots in the frame)
- Camera angle (move the camera slightly between batches)
- Lighting (room light on/off, lamp angle)
- Whether you're fretting notes or open strings
- Different parts of the neck (open position, 5th fret, 12th fret)

Aim for 300–400 images. More variety = better model.

### Day 3–4: Label on Roboflow

1. Create a free account at `roboflow.com`
2. New project → Object Detection → name it `fretboard`
3. Upload your `data/raw_images/` folder
4. Label each image: draw **one bounding box** around the entire fretboard visible in frame. Class name: `fretboard`
5. Label ~250 images minimum (skip blurry/duplicate ones)

**Roboflow labeling tips:**
- Use the keyboard shortcut: draw box → press Enter → next image. Very fast once in rhythm.
- 250 images takes about 60–90 minutes at a steady pace.
- Don't be too precise at edges — within 10–15px is fine.

### Day 5: Export dataset

In Roboflow: Versions → Generate → Add augmentations:
- Flip horizontal: ON
- Brightness: ±25%
- Blur: up to 1.5px
- Mosaic: ON

Export as **YOLOv8 format** → get your API key + version number.

```python
# Run this once to download
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY_HERE")
project = rf.workspace().project("fretboard")
dataset = project.version(1).download("yolov8", location="data/dataset")
```

**Week 3 milestone:** `data/dataset/` contains `train/`, `valid/`, `test/` folders with images and YOLO-format `.txt` label files.

---

## Week 4 — Train YOLOv8n on your M2

### Day 1: Understand your dataset

Before training, inspect it:

```python
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
```

Boxes should tightly surround the fretboard. If they look wrong, fix labels in Roboflow and re-export.

### Day 2: Train

```python
# train.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # downloads ~6MB pretrained weights

results = model.train(
    data="data/dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=8,           # safe for 8GB unified memory
    device="mps",      # Apple GPU
    project="models/runs",
    name="fretboard_v1",
    patience=15,       # stop early if no improvement for 15 epochs
    lr0=0.01,
    augment=True,
    cache=True,        # cache images in RAM for speed
)
```

Run: `python train.py`

**Expected time on M2 Air:** 25–45 minutes for 60 epochs at 640px.  
**What to watch:** `box_loss` and `cls_loss` should trend down. mAP50 should rise above 0.85.

### Day 3: Evaluate results

```bash
# Roboflow auto-saves plots to models/runs/fretboard_v1/
# Open these files:
open models/runs/fretboard_v1/results.png    # loss + mAP curves
open models/runs/fretboard_v1/val_batch0_pred.jpg  # visual predictions
```

If mAP50 < 0.80: add more labeled images (especially the failure cases you found in Week 2) and retrain.

### Day 4: Write the inference wrapper

Create `src/detect.py`:

```python
from ultralytics import YOLO
import numpy as np

class FretboardDetector:
    def __init__(self, model_path="models/runs/fretboard_v1/weights/best.pt"):
        self.model = YOLO(model_path)
        self.last_box = None   # fallback if detection fails in a frame

    def detect(self, frame, conf_threshold=0.5):
        """
        Returns (x1, y1, x2, y2) of the fretboard bounding box,
        or None if not detected this frame.
        """
        results = self.model(frame, verbose=False, device="mps")[0]

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            conf = float(box.conf)
            if conf > conf_threshold and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is not None:
            self.last_box = best_box
            return best_box

        return self.last_box   # return last known if detection drops a frame
```

### Day 5: Test inference speed

```python
# speed_test.py
import cv2, time
from src.detect import FretboardDetector

detector = FretboardDetector()
cap = cv2.VideoCapture(0)
times = []

for _ in range(60):
    ret, frame = cap.read()
    t0 = time.time()
    box = detector.detect(frame)
    times.append(time.time() - t0)
    if box is not None:
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("detect test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print(f"Avg inference: {1000*sum(times)/len(times):.1f}ms")
print(f"Avg FPS: {1/( sum(times)/len(times) ):.1f}")
```

Target: >15 FPS. YOLOv8n on M2 MPS should comfortably hit 25–40 FPS at 640px.

**Week 4 milestone:** `FretboardDetector.detect()` reliably returns a bounding box in real time.

---

## Week 5 — Fret grid mapping + scale overlay

### Day 1–2: Grid math inside the bounding box

Add to `src/grid.py`:

```python
import cv2
import numpy as np

# Standard guitar tuning — open string notes (string 1=high e, 6=low E)
OPEN_NOTES = {
    1: 4,   # e  (MIDI offset from C: E=4)
    2: 11,  # B
    3: 7,   # G
    4: 2,   # D
    5: 9,   # A
    6: 4,   # E
}
# Use chromatic scale index: C=0, C#=1, D=2, D#=3, E=4, F=5,
#                            F#=6, G=7, G#=8, A=9, A#=10, B=11

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

class FretGrid:
    def __init__(self, num_frets=12, num_strings=6):
        self.num_frets = num_frets
        self.num_strings = num_strings

    def compute_grid(self, box):
        """
        Given bounding box (x1,y1,x2,y2), compute pixel positions
        of each fret/string intersection assuming a uniform grid.
        Returns dict: {(string_idx, fret_idx): (px, py)}
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        # Fret spacing — real guitars have decreasing spacing toward body,
        # but uniform is fine for a first version
        fret_xs = [x1 + int(w * f / self.num_frets) for f in range(self.num_frets + 1)]
        string_ys = [y1 + int(h * s / (self.num_strings - 1)) for s in range(self.num_strings)]

        grid = {}
        for s_idx, py in enumerate(string_ys):
            for f_idx, px in enumerate(fret_xs):
                grid[(s_idx + 1, f_idx)] = (px, py)   # 1-indexed strings
        return grid

    def get_note(self, string_idx, fret_idx):
        """Returns chromatic index (0-11) of note at given position."""
        open_note = OPEN_NOTES[string_idx]
        return (open_note + fret_idx) % 12

    def get_note_name(self, string_idx, fret_idx):
        return NOTE_NAMES[self.get_note(string_idx, fret_idx)]
```

### Day 3: Scale definitions

Create `src/scales.py`:

```python
# Intervals from root (semitones)
SCALES = {
    "Major":              [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor":      [0, 2, 3, 5, 7, 8, 10],
    "Pentatonic Major":   [0, 2, 4, 7, 9],
    "Pentatonic Minor":   [0, 3, 5, 7, 10],
    "Blues":              [0, 3, 5, 6, 7, 10],
    "Dorian":             [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian":         [0, 2, 4, 5, 7, 9, 10],
    "Harmonic Minor":     [0, 2, 3, 5, 7, 8, 11],
    "Phrygian":           [0, 1, 3, 5, 7, 8, 10],
    "Lydian":             [0, 2, 4, 6, 7, 9, 11],
    "Locrian":            [0, 1, 3, 5, 6, 8, 10],
    "Whole Tone":         [0, 2, 4, 6, 8, 10],
    "Diminished":         [0, 2, 3, 5, 6, 8, 9, 11],
}

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def get_scale_notes(root_name, scale_name):
    """
    Returns set of chromatic indices in the scale.
    e.g. get_scale_notes('A', 'Pentatonic Minor') → {9, 0, 2, 4, 7}
    """
    root_idx = NOTE_NAMES.index(root_name)
    intervals = SCALES[scale_name]
    return {(root_idx + i) % 12 for i in intervals}, root_idx
```

### Day 4–5: Overlay engine

Create `src/overlay.py`:

```python
import cv2
import numpy as np
from src.grid import FretGrid
from src.scales import get_scale_notes

# Dot colors
COLOR_ROOT     = (0, 80, 255)    # red-orange: root note
COLOR_SCALE    = (0, 220, 255)   # yellow: scale tone
COLOR_INTERVAL = {               # optional: color by interval degree
    0:  (0, 80, 255),    # root
    2:  (0, 200, 255),   # 2nd
    3:  (50, 220, 180),  # minor 3rd
    4:  (50, 220, 180),  # major 3rd
    5:  (100, 255, 100), # 4th
    7:  (0, 180, 255),   # 5th
    9:  (200, 150, 255), # 6th
    10: (200, 100, 255), # minor 7th
    11: (200, 100, 255), # major 7th
}

class ScaleOverlay:
    def __init__(self, num_frets=12):
        self.grid = FretGrid(num_frets=num_frets)
        self.root = 'A'
        self.scale = 'Pentatonic Minor'
        self.show_names = True
        self.dot_radius = 14

    def set_scale(self, root, scale_name):
        self.root = root
        self.scale = scale_name

    def draw(self, frame, box):
        if box is None:
            return frame

        grid_positions = self.grid.compute_grid(box)
        scale_notes, root_idx = get_scale_notes(self.root, self.scale)

        for (string_idx, fret_idx), (px, py) in grid_positions.items():
            note = self.grid.get_note(string_idx, fret_idx)
            if note not in scale_notes:
                continue

            is_root = (note == root_idx)
            color = COLOR_ROOT if is_root else COLOR_SCALE
            r = self.dot_radius + (4 if is_root else 0)

            # Draw filled circle + border
            cv2.circle(frame, (px, py), r, color, -1)
            cv2.circle(frame, (px, py), r, (255,255,255), 1)

            # Note name label
            if self.show_names:
                name = self.grid.get_note_name(string_idx, fret_idx)
                cv2.putText(frame, name, (px - 8, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (20, 20, 20), 1, cv2.LINE_AA)

        # Draw bounding box of detected fretboard
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1,y1),(x2,y2),(100,100,100),1)

        return frame
```

**Week 5 milestone:** Scale dots appear on the fretboard in real time.

---

## Week 6 — Tkinter UI + main loop

### Day 1–2: Control panel

Create `src/ui.py`:

```python
import tkinter as tk
from tkinter import ttk

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
SCALE_NAMES = [
    "Major", "Natural Minor", "Pentatonic Major", "Pentatonic Minor",
    "Blues", "Dorian", "Mixolydian", "Harmonic Minor",
    "Phrygian", "Lydian", "Locrian", "Whole Tone", "Diminished"
]

class ControlPanel:
    def __init__(self, on_change_callback):
        self.callback = on_change_callback
        self.root_var = tk.StringVar(value='A')
        self.scale_var = tk.StringVar(value='Pentatonic Minor')
        self.names_var = tk.BooleanVar(value=True)
        self.frets_var = tk.IntVar(value=12)

        self.win = tk.Tk()
        self.win.title("Scale Mapper")
        self.win.resizable(False, False)
        self._build()

    def _build(self):
        pad = {"padx": 10, "pady": 6}

        tk.Label(self.win, text="Root note").grid(row=0, column=0, **pad, sticky='w')
        ttk.Combobox(self.win, textvariable=self.root_var,
                     values=NOTE_NAMES, width=8, state='readonly') \
            .grid(row=0, column=1, **pad)

        tk.Label(self.win, text="Scale").grid(row=1, column=0, **pad, sticky='w')
        ttk.Combobox(self.win, textvariable=self.scale_var,
                     values=SCALE_NAMES, width=20, state='readonly') \
            .grid(row=1, column=1, **pad)

        tk.Checkbutton(self.win, text="Show note names",
                       variable=self.names_var) \
            .grid(row=2, column=0, columnspan=2, **pad, sticky='w')

        tk.Label(self.win, text="Visible frets").grid(row=3, column=0, **pad, sticky='w')
        tk.Scale(self.win, from_=5, to=22, orient='horizontal',
                 variable=self.frets_var, length=180) \
            .grid(row=3, column=1, **pad)

        tk.Button(self.win, text="Apply", command=self._apply, width=12) \
            .grid(row=4, column=0, columnspan=2, pady=12)

        tk.Label(self.win, text="Press Q in camera window to quit",
                 fg="gray", font=("Arial", 9)) \
            .grid(row=5, column=0, columnspan=2)

    def _apply(self):
        self.callback(
            self.root_var.get(),
            self.scale_var.get(),
            self.names_var.get(),
            self.frets_var.get()
        )

    def update(self):
        self.win.update()   # non-blocking Tkinter tick
```

### Day 3: Main loop

Create `main.py`:

```python
import cv2
from src.camera import get_camera
from src.detect import FretboardDetector
from src.overlay import ScaleOverlay
from src.ui import ControlPanel

def main():
    cap = get_camera()
    detector = FretboardDetector()
    overlay = ScaleOverlay()

    def on_change(root, scale, show_names, num_frets):
        overlay.set_scale(root, scale)
        overlay.show_names = show_names
        overlay.grid.num_frets = num_frets

    panel = ControlPanel(on_change)

    print("Starting. Press Q in camera window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        box = detector.detect(frame)
        frame = overlay.draw(frame, box)

        cv2.imshow("Fretboard Scale Mapper", frame)

        # Tick the Tkinter event loop (non-blocking)
        try:
            panel.update()
        except tk.TclError:
            break   # window was closed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Day 4: Manual calibration fallback

Create `calibrate.py` — click 4 corners yourself when detection fails:

```python
import cv2
import numpy as np
from src.camera import get_camera

corners = []

def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
        print(f"Corner {len(corners)}: ({x}, {y})")

def manual_calibrate():
    cap = get_camera()
    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", click_corner)

    print("Click the 4 corners of the fretboard (top-left, top-right, bottom-right, bottom-left)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, (cx, cy) in enumerate(corners):
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (cx+8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        if len(corners) == 4:
            pts = np.array(corners, dtype=np.float32)
            x_min = int(min(p[0] for p in corners))
            y_min = int(min(p[1] for p in corners))
            x_max = int(max(p[0] for p in corners))
            y_max = int(max(p[1] for p in corners))
            box = (x_min, y_min, x_max, y_max)
            print(f"\nManual box: {box}")
            print("Press S to save, R to reset")

        msg = f"Corners: {len(corners)}/4"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Calibrate", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            corners.clear()
        elif key == ord('s') and len(corners) == 4:
            np.save("calibration.npy", np.array(corners))
            print("Saved to calibration.npy")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_calibrate()
```

### Day 5: Wire calibration into detector

In `src/detect.py`, add:

```python
import os
import numpy as np

class FretboardDetector:
    def __init__(self, model_path="..."):
        self.model = YOLO(model_path)
        self.last_box = None
        self._load_calibration()

    def _load_calibration(self):
        if os.path.exists("calibration.npy"):
            pts = np.load("calibration.npy")
            xs = pts[:,0]; ys = pts[:,1]
            self.calibrated_box = (int(xs.min()), int(ys.min()),
                                   int(xs.max()), int(ys.max()))
            print("Loaded calibration fallback")
        else:
            self.calibrated_box = None

    def detect(self, frame, conf_threshold=0.5):
        # ... existing detection code ...
        if best_box is None:
            return self.last_box or self.calibrated_box
        self.last_box = best_box
        return best_box
```

**Week 6 milestone:** Full app runs end-to-end. Tkinter panel changes scale in real time.

---

## Week 7 — Robustness + real-world fixes

### Temporal smoothing (stop jitter)

Add to `src/detect.py`:

```python
from collections import deque

class FretboardDetector:
    def __init__(self, ...):
        ...
        self.history = deque(maxlen=3)   # average over last 3 frames

    def detect(self, frame, conf_threshold=0.5):
        # ... get best_box as before ...
        if best_box is not None:
            self.history.append(best_box)

        if not self.history:
            return self.calibrated_box

        # Average the box coordinates
        arr = np.array(self.history)
        smoothed = arr.mean(axis=0).astype(int)
        return tuple(smoothed)
```

### Non-uniform fret spacing

Real guitars have closer frets toward the body. Add to `FretGrid`:

```python
def fret_positions_real(self, x1, x2, num_frets):
    """
    Use equal-temperament fret ratios.
    Each fret is 1/17.817 of the remaining scale length from that fret.
    """
    scale = x2 - x1
    positions = [x1]
    remaining = scale
    x = x1
    for _ in range(num_frets):
        step = remaining / 17.817
        x += step
        remaining -= step
        positions.append(int(x))
    return positions
```

### Adaptive thresholding for different lighting

In `src/grid.py` preprocessing (used in classical fallback):

```python
def preprocess_adaptive(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE: improves contrast in uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    return edges
```

### Performance profiling

```python
# Add to main loop to track bottlenecks
import time

t0 = time.perf_counter()
box = detector.detect(frame)
t1 = time.perf_counter()
frame = overlay.draw(frame, box)
t2 = time.perf_counter()

detect_ms = (t1-t0)*1000
draw_ms   = (t2-t1)*1000
cv2.putText(frame, f"detect:{detect_ms:.0f}ms draw:{draw_ms:.0f}ms",
            (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
```

**Week 7 milestone:** App runs smoothly in variable lighting. Box doesn't jump around.

---

## Week 8 — Stretch goals (pick what interests you)

### Save a screenshot
```python
# In camera loop, press S:
if key == ord('s'):
    cv2.imwrite(f"screenshot_{int(time.time())}.png", frame)
```

### Color dots by interval
Instead of a flat color for scale tones, use `COLOR_INTERVAL` dict from `overlay.py`:
```python
interval = (note - root_idx) % 12
color = COLOR_INTERVAL.get(interval, COLOR_SCALE)
```

### Display scale name on frame
```python
label = f"{overlay.root} {overlay.scale}"
cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
```

### Add chord shape detection (advanced)
If you want to detect which chord is being fretted, you'd need a second model or a hand landmark detector (MediaPipe Hands — free, no training required). This is a meaningful extension project on its own.

---

## Requirements file

```
# requirements.txt
opencv-python>=4.9
numpy>=1.26
ultralytics>=8.2
roboflow>=1.1
pillow>=10.0
```

Install everything: `pip install -r requirements.txt`

---

## Common problems and fixes

| Problem | Likely cause | Fix |
|---|---|---|
| MPS not available | Wrong PyTorch build | Reinstall: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` then check again |
| Training crashes OOM | Batch too large | Set `batch=4` in train() |
| Detection is slow (<10 FPS) | Running on CPU | Confirm `device="mps"` in detect() call |
| Grid dots wildly off | Bounding box wrong | Run calibrate.py and use manual corners |
| Box jitters constantly | No smoothing | Increase `deque(maxlen=N)` — try N=12 |
| Roboflow export fails | API key wrong | Check in Roboflow dashboard → Settings → API |
| Lines not detected in Week 2 | Hough threshold too high | Lower threshold from 80 → 40, check edges image |

---

## Key things to write in your notes.md

After each week, record:
- What thresholds worked for your lighting
- Which scale felt hardest to overlay correctly and why
- Training mAP50 score achieved
- Average inference FPS
- Any data augmentation that made a noticeable difference

This becomes invaluable when you revisit the project in 3 months.
