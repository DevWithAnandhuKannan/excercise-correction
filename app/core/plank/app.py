import os
import cv2
import numpy as np
import pandas as pd
import joblib  # Changed from pickle
import mediapipe as mp
from flask import Flask, Response, render_template_string
import math

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_important_keypoints(results, important_landmarks: list) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def get_static_file_url(file_name: str) -> str:
    path = os.path.join(os.getcwd(), file_name)
    return path if os.path.exists(path) else None

def get_drawing_color(error: bool) -> tuple:
    LIGHT_BLUE = (244, 117, 66)
    LIGHT_PINK = (245, 66, 230)
    LIGHT_RED = (29, 62, 199)
    LIGHT_YELLOW = (1, 143, 241)
    return (LIGHT_YELLOW, LIGHT_RED) if error else (LIGHT_BLUE, LIGHT_PINK)

class PlankDetection:
    ML_MODEL_PATH = get_static_file_url("model/plank_model.pkl")
    INPUT_SCALER_PATH = get_static_file_url("model/plank_input_scaler.pkl")
    PREDICTION_PROBABILITY_THRESHOLD = 0.6

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False

    def init_important_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
            "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        ]
        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def load_machine_learning_model(self) -> None:
        if not self.ML_MODEL_PATH or not self.INPUT_SCALER_PATH:
            raise Exception("Cannot found plank model file or input scaler file")
        try:
            with open(self.ML_MODEL_PATH, "rb") as f:
                self.model = joblib.load(f)  # Changed to joblib
            with open(self.INPUT_SCALER_PATH, "rb") as f2:
                self.input_scaler = joblib.load(f2)  # Changed to joblib
            print(f"Loading scaler from: {self.INPUT_SCALER_PATH}")
            print(f"Scaler type: {type(self.input_scaler)}")
            print(f"Has 'transform' method: {hasattr(self.input_scaler, 'transform')}")
            if not hasattr(self.input_scaler, 'transform'):
                raise AttributeError("Loaded scaler does not have 'transform' method")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def handle_detected_results(self, video_name: str) -> None:
        file_name, _ = video_name.split(".")
        save_folder = get_static_file_url("images")
        for index, error in enumerate(self.results):
            try:
                image_name = f"{file_name}_{index}.jpg"
                cv2.imwrite(f"{save_folder}/{file_name}_{index}.jpg", error["frame"])
                self.results[index]["frame"] = image_name
            except Exception as e:
                print("ERROR cannot save frame: " + str(e))
                self.results[index]["frame"] = None
        return self.results, self.previous_stage

    def clear_results(self) -> None:
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False

    def detect(self, mp_results, image, timestamp) -> None:
        try:
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))
            predicted_class = self.model.predict(X)[0]
            prediction_probability = self.model.predict_proba(X)[0]

            if (predicted_class == "C" and 
                prediction_probability[prediction_probability.argmax()] >= self.PREDICTION_PROBABILITY_THRESHOLD):
                current_stage = "correct"
            elif (predicted_class == "L" and 
                  prediction_probability[prediction_probability.argmax()] >= self.PREDICTION_PROBABILITY_THRESHOLD):
                current_stage = "low back"
            elif (predicted_class == "H" and 
                  prediction_probability[prediction_probability.argmax()] >= self.PREDICTION_PROBABILITY_THRESHOLD):
                current_stage = "high back"
            else:
                current_stage = "unknown"

            if current_stage in ["low back", "high back"]:
                if self.previous_stage == current_stage:
                    pass
                elif self.previous_stage != current_stage:
                    self.results.append({"stage": current_stage, "frame": image.copy(), "timestamp": timestamp})
                    self.has_error = True
            else:
                self.has_error = False

            self.previous_stage = current_stage

            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1)
            )

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability[np.argmax(prediction_probability)], 2)),
                        (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            raise Exception(f"Error while detecting plank errors: {e}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

plank_detector = PlankDetection()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            plank_detector.detect(results, frame_bgr, timestamp=0)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Plank Detector</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            h1 { color: #333; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Real-Time Plank Detector</h1>
        <img src="{{ url_for('video_feed') }}" alt="Video Feed">
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=3000, debug=True)
    finally:
        cap.release()
        pose.close()