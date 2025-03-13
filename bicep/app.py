import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from flask import Flask, Response, render_template_string
import math
import traceback

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Utility functions
def calculate_angle(pointA: list, pointB: list, pointC: list) -> float:
    """Calculate angle between three points."""
    a = np.array(pointA)
    b = np.array(pointB)
    c = np.array(pointC)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

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

# BicepPoseAnalysis class (unchanged from your code)
class BicepPoseAnalysis:
    def __init__(
        self,
        side: str,
        stage_down_threshold: float,
        stage_up_threshold: float,
        peak_contraction_threshold: float,
        loose_upper_arm_angle_threshold: float,
        visibility_threshold: float,
    ):
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold
        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000

    def get_joints(self, landmarks) -> bool:
        side = self.side.upper()
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
        ]
        self.is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        if not self.is_visible:
            return self.is_visible
        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
        ]
        self.elbow = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y,
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
        ]
        return self.is_visible

    def analyze_pose(self, landmarks, frame, results, timestamp: int, lean_back_error: bool = False):
        has_error = False
        self.get_joints(landmarks)
        if not self.is_visible:
            return (None, None, has_error)
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))
        if lean_back_error:
            return (bicep_curl_angle, ground_upper_arm_angle, has_error)
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            has_error = True
            cv2.rectangle(frame, (350, 0), (600, 40), (245, 117, 16), -1)
            cv2.putText(frame, "ARM ERROR", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "LOOSE UPPER ARM", (355, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
                results.append({"stage": "loose upper arm", "frame": frame.copy(), "timestamp": timestamp})
        else:
            self.loose_upper_arm = False
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle
        elif self.stage == "down":
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                cv2.rectangle(frame, (350, 0), (600, 40), (245, 117, 16), -1)
                cv2.putText(frame, "ARM ERROR", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "WEAK PEAK CONTRACTION", (355, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                self.detected_errors["PEAK_CONTRACTION"] += 1
                results.append({"stage": "peak contraction", "frame": frame.copy(), "timestamp": timestamp})
                has_error = True
            self.peak_contraction_angle = 1000
        return (bicep_curl_angle, ground_upper_arm_angle, has_error)

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000

# BicepCurlDetection class
class BicepCurlDetection:
    ML_MODEL_PATH = get_static_file_url("model/bicep_curl_model.pkl")
    INPUT_SCALER = get_static_file_url("model/bicep_curl_input_scaler.pkl")
    VISIBILITY_THRESHOLD = 0.65
    STAGE_UP_THRESHOLD = 100
    STAGE_DOWN_THRESHOLD = 120
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
    POSTURE_ERROR_THRESHOLD = 0.95

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()
        self.left_arm_analysis = BicepPoseAnalysis(
            side="left",
            stage_down_threshold=self.STAGE_DOWN_THRESHOLD,
            stage_up_threshold=self.STAGE_UP_THRESHOLD,
            peak_contraction_threshold=self.PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=self.LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=self.VISIBILITY_THRESHOLD,
        )
        self.right_arm_analysis = BicepPoseAnalysis(
            side="right",
            stage_down_threshold=self.STAGE_DOWN_THRESHOLD,
            stage_up_threshold=self.STAGE_UP_THRESHOLD,
            peak_contraction_threshold=self.PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=self.LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=self.VISIBILITY_THRESHOLD,
        )
        self.stand_posture = 0
        self.previous_stand_posture = 0
        self.results = []
        self.has_error = False

    def init_important_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
        ]
        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def load_machine_learning_model(self) -> None:
        if not self.ML_MODEL_PATH or not self.INPUT_SCALER:
            raise Exception("Cannot find bicep curl model or scaler file")
        try:
            with open(self.ML_MODEL_PATH, "rb") as f:
                self.model = joblib.load(f)
            with open(self.INPUT_SCALER, "rb") as f2:
                self.input_scaler = joblib.load(f2)
            print(f"Loading scaler from: {self.INPUT_SCALER}")
            print(f"Scaler type: {type(self.input_scaler)}")
            print(f"Has 'transform' method: {hasattr(self.input_scaler, 'transform')}")
            if not hasattr(self.input_scaler, 'transform'):
                raise AttributeError("Loaded scaler does not have 'transform' method")
        except Exception as e:
            raise Exception(f"Error loading model or scaler: {e}")

    def handle_detected_results(self, video_name: str) -> tuple:
        file_name, _ = video_name.split(".")
        save_folder = get_static_file_url("images")
        os.makedirs(save_folder, exist_ok=True)
        for index, error in enumerate(self.results):
            try:
                image_name = f"{file_name}_{index}.jpg"
                cv2.imwrite(f"{save_folder}/{image_name}", error["frame"])
                self.results[index]["frame"] = image_name
            except Exception as e:
                print("ERROR cannot save frame: " + str(e))
                self.results[index]["frame"] = None
        return self.results, {
            "left_counter": self.left_arm_analysis.get_counter(),
            "right_counter": self.right_arm_analysis.get_counter(),
        }

    def clear_results(self) -> None:
        self.stand_posture = 0
        self.previous_stand_posture = 0
        self.results = []
        self.has_error = False
        self.right_arm_analysis.reset()
        self.left_arm_analysis.reset()

    def detect(self, mp_results, image, timestamp: int) -> None:
        self.has_error = False
        try:
            video_dimensions = [image.shape[1], image.shape[0]]
            landmarks = mp_results.pose_landmarks.landmark
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))
            predicted_class = self.model.predict(X)[0]
            prediction_probabilities = self.model.predict_proba(X)[0]
            class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)
            if class_prediction_probability >= self.POSTURE_ERROR_THRESHOLD:
                self.stand_posture = predicted_class
            if self.stand_posture == "L":
                if self.previous_stand_posture != self.stand_posture:
                    self.results.append({"stage": "lean too far back", "frame": image.copy(), "timestamp": timestamp})
                self.has_error = True
            self.previous_stand_posture = self.stand_posture
            (left_bicep_curl_angle, left_ground_upper_arm_angle, left_arm_error) = self.left_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image, results=self.results, timestamp=timestamp, lean_back_error=(self.stand_posture == "L")
            )
            (right_bicep_curl_angle, right_ground_upper_arm_angle, right_arm_error) = self.right_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image, results=self.results, timestamp=timestamp, lean_back_error=(self.stand_posture == "L")
            )
            self.has_error = True if (right_arm_error or left_arm_error or self.has_error) else False
            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1)
            )
            cv2.rectangle(image, (0, 0), (350, 40), (245, 117, 16), -1)
            cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.right_arm_analysis.counter) if self.right_arm_analysis.is_visible else "UNK",
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.left_arm_analysis.counter) if self.left_arm_analysis.is_visible else "UNK",
                        (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Lean-Too-Far-Back", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"{'ERROR' if self.stand_posture == 'L' else 'CORRECT'}, {predicted_class}, {class_prediction_probability}",
                        (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.left_arm_analysis.is_visible:
                cv2.putText(image, str(left_bicep_curl_angle),
                            tuple(np.multiply(self.left_arm_analysis.elbow, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_ground_upper_arm_angle),
                            tuple(np.multiply(self.left_arm_analysis.shoulder, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.right_arm_analysis.is_visible:
                cv2.putText(image, str(right_bicep_curl_angle),
                            tuple(np.multiply(self.right_arm_analysis.elbow, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_ground_upper_arm_angle),
                            tuple(np.multiply(self.right_arm_analysis.shoulder, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            traceback.print_exc()
            raise e

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

# Initialize detector
bicep_detector = BicepCurlDetection()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            bicep_detector.detect(results, frame_bgr, timestamp=0)
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
        <title>Bicep Curl Detector</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            h1 { color: #333; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Real-Time Bicep Curl Detector</h1>
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