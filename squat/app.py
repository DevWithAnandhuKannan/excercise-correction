import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from flask import Flask, Response, render_template_string
from collections import deque
import math

# Initialize Flask app
app = Flask(__name__)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load model and scaler
model = joblib.load("model/squat_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Utility functions
def calculate_distance(pointX: list, pointY: list) -> float:
    x1, y1 = pointX
    x2, y2 = pointY
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_important_keypoints(results, important_landmarks: list) -> list:
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def get_drawing_color(error: bool) -> tuple:
    LIGHT_BLUE = (244, 117, 66)    # Correct: Light Blue
    LIGHT_PINK = (245, 66, 230)    # Correct: Light Pink
    LIGHT_RED = (29, 62, 199)      # Error: Light Red
    LIGHT_YELLOW = (1, 143, 241)   # Error: Light Yellow
    return (LIGHT_YELLOW, LIGHT_RED) if error else (LIGHT_BLUE, LIGHT_PINK)

def get_static_file_url(file_name: str) -> str:
    path = os.path.join(os.getcwd(), file_name)
    return path if os.path.exists(path) else None

# Squat detection parameters
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]
HEADERS = ["label"] + [f"{lm.lower()}_{s}" for lm in IMPORTANT_LMS for s in ["x", "y", "z", "v"]]
PREDICTION_PROB_THRESHOLD = 0.7
VISIBILITY_THRESHOLD = 0.6
FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down": [0.7, 1.1],
}

# Squat counter variables
squat_count = 0
current_stage = ""
state_buffer = deque(maxlen=5)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

def analyze_foot_knee_placement(results, stage, foot_shoulder_ratio_thresholds, knee_foot_ratio_thresholds, visibility_threshold):
    analyzed_results = {"foot_placement": -1, "knee_placement": -1}
    landmarks = results.pose_landmarks.landmark

    # Visibility check
    vis_checks = [
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
    ]
    if any(v < visibility_threshold for v in vis_checks):
        return analyzed_results

    # Calculate distances
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
    right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
    foot_width = calculate_distance(left_foot, right_foot)

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    knee_width = calculate_distance(left_knee, right_knee)

    # Ratios
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # Foot placement
    min_ratio, max_ratio = foot_shoulder_ratio_thresholds
    if min_ratio <= foot_shoulder_ratio <= max_ratio:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio:
        analyzed_results["foot_placement"] = 2

    # Knee placement
    if stage:
        min_ratio, max_ratio = knee_foot_ratio_thresholds.get(stage, [0, float('inf')])
        if min_ratio <= knee_foot_ratio <= max_ratio:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < min_ratio:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > max_ratio:
            analyzed_results["knee_placement"] = 2

    return analyzed_results

def generate_frames():
    global squat_count, current_stage
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Model prediction
            row = extract_important_keypoints(results, IMPORTANT_LMS)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X_scaled = scaler.transform(X)
            predicted_class = model.predict(X_scaled)[0]
            prediction_prob = model.predict_proba(X_scaled)[0].max()

            # Stage logic
            if predicted_class == "down" and prediction_prob >= PREDICTION_PROB_THRESHOLD:
                current_stage = "down"
            elif current_stage == "down" and predicted_class == "up" and prediction_prob >= PREDICTION_PROB_THRESHOLD:
                current_stage = "up"
                squat_count += 1

            # Analyze placement
            analyzed_results = analyze_foot_knee_placement(
                results, current_stage, FOOT_SHOULDER_RATIO_THRESHOLDS, KNEE_FOOT_RATIO_THRESHOLDS, VISIBILITY_THRESHOLD
            )

            # Interpret results
            foot_placement = {0: "correct", 1: "too tight", 2: "too wide", -1: "unknown"}[analyzed_results["foot_placement"]]
            knee_placement = {0: "correct", 1: "too tight", 2: "too wide", -1: "unknown"}[analyzed_results["knee_placement"]]
            has_error = foot_placement in ["too tight", "too wide"] or knee_placement in ["too tight", "too wide"]

            # Visualization
            landmark_color, connection_color = get_drawing_color(has_error)
            mp_drawing.draw_landmarks(
                frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1)
            )

            # Status box
            cv2.rectangle(frame_bgr, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame_bgr, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(frame_bgr, f"{squat_count}, {predicted_class}, {prediction_prob:.2f}", (5, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_bgr, "FEET", (130, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(frame_bgr, foot_placement, (125, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_bgr, "KNEE", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(frame_bgr, knee_placement, (220, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

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
        <title>Squat Counter</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            h1 { color: #333; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Real-Time Squat Counter</h1>
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