from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import cv2
import logging
import face_recognition
import os
import json
import time
import numpy as np
from werkzeug.utils import secure_filename
import threading
from database import get_logs

logging.basicConfig(level=logging.DEBUG)

CAPTURED_FACES_DIR = "captured_faces"
KNOWN_FACES_DIR = "known_faces"
os.makedirs(CAPTURED_FACES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = KNOWN_FACES_DIR

AVAILABLE_COLORS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Purple": (255, 0, 255),
    "Cyan": (0, 255, 255),
    "Maroon": (128, 0, 0),
    "Olive": (128, 128, 0),
    "Navy": (0, 0, 128)
}

used_colors = {}
CATEGORY_FILE = "categories.json"
if os.path.exists(CATEGORY_FILE):
    with open(CATEGORY_FILE, "r") as f:
        used_colors = json.load(f)
else:
    used_colors = {}

class FaceDetector:
    def __init__(self, pipeline, cooldown_time=5):
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            logging.error("Unable to access the camera stream via GStreamer.")
            raise RuntimeError("Unable to access the camera stream.")

        cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.cooldown_time = cooldown_time
        self.recent_faces = {}
        self.known_faces, self.known_names, self.category_colors = self.load_known_faces()
        self.last_alert_time = 0
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def load_known_faces(self):
        known_faces = []
        known_names = []
        category_colors = {}

        # Load color config once and respect saved values
        with open(CATEGORY_FILE, "r") as f:
            color_config = json.load(f)

        for category in os.listdir(KNOWN_FACES_DIR):
            category_path = os.path.join(KNOWN_FACES_DIR, category)
            if os.path.isdir(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith((".jpg", ".png")):
                        path = os.path.join(category_path, filename)
                        image = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_faces.append(encodings[0])
                            known_names.append((category, filename.split(".")[0]))

                # Assign colors properly from config
                color_name = color_config.get(category)
                if color_name is None:
                    color_index = len(color_config) % len(AVAILABLE_COLORS)
                    color_name = list(AVAILABLE_COLORS.keys())[color_index]
                    color_config[category] = color_name
                    with open(CATEGORY_FILE, 'w') as f:
                        json.dump(color_config, f, indent=4)

                category_colors[category] = AVAILABLE_COLORS[color_name]

        logging.info(f"Loaded {len(known_faces)} known faces across {len(category_colors)} categories.")
        return known_faces, known_names, category_colors


    def detect_and_recognize_faces(self):
        self.running = True
        try:
            while self.running:
                self.cap.grab()
                ret, frame = self.cap.retrieve()

                if not ret:
                    time.sleep(0.1)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    face_image = frame[y:y+h, x:x+w]
                    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_face)

                    if len(encodings) == 0:
                        continue

                    face_encoding = encodings[0]
                    matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.5) if self.known_faces else []

                    category, name = "Unknown", "Unknown"
                    if True in matches:
                        match_index = matches.index(True)
                        category, name = self.known_names[match_index]
                        timestamp = int(time.time())
                        save_path = os.path.join(CAPTURED_FACES_DIR, category, f"{name}_{timestamp}.jpg")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, face_image)

                    rect_color = self.category_colors.get(category, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
                    cv2.putText(frame, f"{category}: {name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

                with self.frame_lock:
                    self.latest_frame = frame.copy()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            if self.cap:
                self.cap.release()  # For OpenCV, releases GStreamer pipeline too
                # Force a delay to allow underlying GStreamer to clean up properly
                time.sleep(0.2)
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logging.warning(f"OpenCV window cleanup issue: {e}")


    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            time.sleep(0.2)
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.warning(f"OpenCV cleanup failed: {e}")


face_detector = None

def start_detection():
    global face_detector
    if face_detector is not None:
        return

    ip = "192.168.0.173"
    username = "admin"
    password = "123456"
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=1"

    pipeline = (
        f"rtspsrc location={rtsp_url} latency=30 ! "
        "rtph265depay ! h265parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop = 1 max-buffers=1 sync=false"
)


    try:
        face_detector = FaceDetector(pipeline)
        threading.Thread(target=face_detector.detect_and_recognize_faces, daemon=True).start()
    except RuntimeError as e:
        logging.error("Start detection failed: " + str(e))
        face_detector = None

@app.route('/')
def index():
    global face_detector
    if face_detector is None:
        try:
            start_detection()
        except RuntimeError as e:
            return "Camera stream could not be started. Check RTSP URL or camera access.", 500
    return render_template('index.html', used_colors=used_colors)

def gen_frames():
    global face_detector
    while True:
        if face_detector is None:
            time.sleep(0.1)
            continue
        with face_detector.frame_lock:
            frame = face_detector.latest_frame.copy() if face_detector.latest_frame is not None else None
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    def stream_safe():
        while face_detector is None or getattr(face_detector, "cap", None) is None:
            time.sleep(0.1)
        yield from gen_frames()
    return app.response_class(stream_safe(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    start_detection()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        category = request.form.get('category')
        if not category:
            return "Category is required", 400
        category_path = os.path.join(KNOWN_FACES_DIR, category)
        os.makedirs(category_path, exist_ok=True)
        file.save(os.path.join(category_path, filename))
        return redirect(url_for('index'))

@app.route('/categories', methods=['GET', 'POST'])
def manage_categories():
    if request.method == 'POST':
        category = request.form.get('category')
        color = request.form.get('color')
        if not category or not color:
            return "Category and color are required", 400
        if category in used_colors:
            return "Category already exists", 400
        used_colors[category] = color
        with open(CATEGORY_FILE, 'w') as f:
            json.dump(used_colors, f, indent=4)
        category_path = os.path.join(KNOWN_FACES_DIR, category)
        os.makedirs(category_path, exist_ok=True)
        return redirect(url_for('manage_categories'))
    return render_template('categories.html', categories=used_colors, colors=AVAILABLE_COLORS)

@app.route('/delete_category', methods=['POST'])
def delete_category():
    category = request.form.get('category')
    if not category:
        return "Category is required", 400
    if category not in used_colors:
        return "Category does not exist", 400
    if category == "Barred":
        return "Cannot delete the 'Barred' category", 400
    del used_colors[category]
    with open(CATEGORY_FILE, 'w') as f:
        json.dump(used_colors, f, indent=4)
    category_path = os.path.join(KNOWN_FACES_DIR, category)
    if os.path.exists(category_path):
        for file in os.listdir(category_path):
            os.remove(os.path.join(category_path, file))
        os.rmdir(category_path)
    return redirect(url_for('manage_categories'))
@app.route('/remove_image/<category>/<filename>', methods=['POST'])
def remove_image(category, filename):
    """Remove an image from the specified category."""
    try:
        image_path = os.path.join(KNOWN_FACES_DIR if category in os.listdir(KNOWN_FACES_DIR) else CAPTURED_FACES_DIR, category, filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"Removed {image_path}")
            return redirect(url_for('view_known_visitors' if category in os.listdir(KNOWN_FACES_DIR) else 'view_previous_visitors'))
        else:
            return "File not found.", 404
    except Exception as e:
        logging.error(f"Failed to remove image: {e}")
        return "An error occurred while removing the image.", 500

@app.route('/move_to_known/<category>/<filename>', methods=['POST'])
def move_to_known(category, filename):
    """Move an image from previous visitors to known visitors."""
    try:
        src_path = os.path.join(CAPTURED_FACES_DIR, category, filename)
        dest_path = os.path.join(KNOWN_FACES_DIR, category, filename)
        
        # Ensure the destination directory exists
        os.makedirs(os.path.join(KNOWN_FACES_DIR, category), exist_ok=True)
        
        # Move the file
        os.rename(src_path, dest_path)
        logging.info(f"Moved {src_path} to {dest_path}")
        return redirect(url_for('view_previous_visitors'))
    except Exception as e:
        logging.error(f"Failed to move image: {e}")
        return "An error occurred while moving the image.", 500
@app.route('/view_previous_visitors')
def view_previous_visitors():
    images = []
    for category in os.listdir(CAPTURED_FACES_DIR):
        category_path = os.path.join(CAPTURED_FACES_DIR, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    images.append((category, filename))
    return render_template('view_previous_visitors.html', images=images)

@app.route('/view_known_visitors')
def view_known_visitors():
    images = []
    for category in os.listdir(KNOWN_FACES_DIR):
        category_path = os.path.join(KNOWN_FACES_DIR, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    images.append((category, filename))
    return render_template('view_known_visitors.html', images=images)

@app.route('/images/<category>/<filename>')
def get_image(category, filename):
    return send_from_directory(os.path.join(KNOWN_FACES_DIR, category), filename)

@app.route('/log_visit', methods=['POST'])
def log_visit_route():
    """Route to log a visitor when detected."""
    name = request.form.get('name')
    category = request.form.get('category')
    if not name or not category:
        return "Name and category are required", 400
    log_visit(name, category)
    return redirect(url_for('view_logs'))

@app.route('/logs', methods=['GET'])
def view_logs():
    """Route to view visitor logs with optional filters."""
    name = request.args.get('name')
    category = request.args.get('category')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    logs = get_logs(name, category, start_date, end_date)
    return render_template('logs.html', logs=logs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
