import cv2
import easyocr
import re
import os
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response

# ================= CONFIG =================
YOLO_MODEL = "D:\\app\\npdataset\\runs\\detect\\runs\\license_plate_yolov8\\weights\\best.pt"
CAMERA_ID = 0
CONF_THRES = 0.3
SAVE_DIR = "detections"
FRAME_SKIP = 1              # keep 1 for stability
HOLD_FRAMES = 12            # box persistence
# =========================================

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD MODELS =================
model = YOLO(YOLO_MODEL)
reader = easyocr.Reader(['en'], gpu=False)

PLATE_REGEX = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}")

app = Flask(__name__)

# ================= GLOBAL STATE =================
last_tracks = {}
frame_count = 0

# ================= PREPROCESS =================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return th

# ================= OCR =================
def read_plate(img):
    result = reader.readtext(preprocess(img))
    best_text, best_conf = None, 0

    for _, text, conf in result:
        text = text.replace(" ", "").upper()
        if PLATE_REGEX.match(text) and conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, best_conf

# ================= CAMERA STREAM =================
def gen_frames():
    global last_tracks, frame_count

    cap = cv2.VideoCapture(CAMERA_ID)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---- YOLO TRACKING (KEY FIX) ----
        results = model.track(
            frame,
            conf=CONF_THRES,
            persist=True,
            verbose=False
        )

        current_ids = set()

        for r in results:
            if r.boxes.id is None:
                continue

            for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                plate = frame[y1:y2, x1:x2]
                if plate.size == 0:
                    continue

                text, conf = read_plate(plate)

                # Store/update track
                last_tracks[track_id] = {
                    "box": (x1, y1, x2, y2),
                    "text": text,
                    "last_seen": frame_count
                }

                current_ids.add(track_id)

                # Save once when OCR succeeds
                if text:
                    filename = f"{text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
                    print(f"✅ Plate detected: {text}")

        # ---- DRAW STABLE BOXES ----
        for track_id, data in list(last_tracks.items()):
            if frame_count - data["last_seen"] > HOLD_FRAMES:
                del last_tracks[track_id]
                continue

            x1, y1, x2, y2 = data["box"]
            label = data["text"] if data["text"] else f"Plate #{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # ---- STREAM TO BROWSER ----
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame + b"\r\n"
        )

    cap.release()

# ================= FLASK =================
@app.route("/")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= RUN =================
if __name__ == "__main__":
    print("🌐 Open browser: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
