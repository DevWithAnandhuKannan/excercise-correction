import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from app.core.exercise_detection import ExerciseDetection
from collections import deque
import math

# Define module-level Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class SquatDetection(ExerciseDetection):
    PREDICTION_PROB_THRESHOLD = 0.7
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down": [0.7, 1.1],
    }

    def __init__(self, app):
        super().__init__(app, "squat/model", "squat_model.pkl", "scaler.pkl")
        self.stages = {
            "up": "up",
            "down": "down",
            "middle": "middle"  # Optional intermediate stage if your model supports it
        }
        self.squat_count = 0
        self.current_stage = ""
        self.state_buffer = deque(maxlen=5)
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False
        self.init_squat_landmarks()

    def init_squat_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
        ]
        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def calculate_distance(self, pointX: list, pointY: list) -> float:
        x1, y1 = pointX
        x2, y2 = pointY
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def analyze_foot_knee_placement(self, results, stage):
        analyzed_results = {"foot_placement": -1, "knee_placement": -1}
        landmarks = results.pose_landmarks.landmark

        # Visibility check
        vis_checks = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
        ]
        if any(v < self.VISIBILITY_THRESHOLD for v in vis_checks):
            return analyzed_results

        # Calculate distances
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)

        left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        foot_width = self.calculate_distance(left_foot, right_foot)

        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        knee_width = self.calculate_distance(left_knee, right_knee)

        # Ratios
        foot_shoulder_ratio = round(foot_width / shoulder_width, 1)
        knee_foot_ratio = round(knee_width / foot_width, 1)

        # Foot placement
        min_ratio, max_ratio = self.FOOT_SHOULDER_RATIO_THRESHOLDS
        if min_ratio <= foot_shoulder_ratio <= max_ratio:
            analyzed_results["foot_placement"] = 0
        elif foot_shoulder_ratio < min_ratio:
            analyzed_results["foot_placement"] = 1
        elif foot_shoulder_ratio > max_ratio:
            analyzed_results["foot_placement"] = 2

        # Knee placement
        if stage:
            min_ratio, max_ratio = self.KNEE_FOOT_RATIO_THRESHOLDS.get(stage, [0, float('inf')])
            if min_ratio <= knee_foot_ratio <= max_ratio:
                analyzed_results["knee_placement"] = 0
            elif knee_foot_ratio < min_ratio:
                analyzed_results["knee_placement"] = 1
            elif knee_foot_ratio > max_ratio:
                analyzed_results["knee_placement"] = 2

        return analyzed_results

    def detect(self, mp_results, image, timestamp, stages: dict):
        try:
            if not self.model or not self.input_scaler:
                cv2.putText(image, "Model not loaded", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return

            # Model prediction
            row = self.extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X_scaled = pd.DataFrame(self.input_scaler.transform(X))
            predicted_class = self.model.predict(X_scaled)[0]
            prediction_prob = self.model.predict_proba(X_scaled)[0].max()

            # Stage logic
            if predicted_class == "down" and prediction_prob >= self.PREDICTION_PROB_THRESHOLD:
                current_stage = "down"
            elif self.current_stage == "down" and predicted_class == "up" and prediction_prob >= self.PREDICTION_PROB_THRESHOLD:
                current_stage = "up"
                self.squat_count += 1
            else:
                current_stage = self.current_stage or "up"  # Default to "up" if no transition

            self.state_buffer.append(current_stage)
            self.current_stage = max(set(self.state_buffer), key=self.state_buffer.count)  # Majority vote

            # Analyze placement
            analyzed_results = self.analyze_foot_knee_placement(mp_results, self.current_stage)
            foot_placement = {0: "correct", 1: "too tight", 2: "too wide", -1: "unknown"}[analyzed_results["foot_placement"]]
            knee_placement = {0: "correct", 1: "too tight", 2: "too wide", -1: "unknown"}[analyzed_results["knee_placement"]]
            has_error = foot_placement in ["too tight", "too wide"] or knee_placement in ["too tight", "too wide"]

            if has_error and self.previous_stage != self.current_stage:
                self.results.append({"stage": self.current_stage, "frame": image.copy(), "timestamp": timestamp, 
                                     "foot_placement": foot_placement, "knee_placement": knee_placement})
                self.has_error = True
            else:
                self.has_error = False

            self.previous_stage = self.current_stage

            # Visualization
            landmark_color, connection_color = self.get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1)
            )

            # Status box
            cv2.rectangle(image, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(image, f"{self.squat_count}, {self.current_stage}, {prediction_prob:.2f}", (5, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "FEET", (130, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(image, foot_placement, (125, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "KNEE", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(image, knee_placement, (220, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(f"Error while detecting squat errors: {e}")
            raise