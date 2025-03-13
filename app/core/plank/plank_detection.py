import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp  # Import mediapipe
from app.core.exercise_detection import ExerciseDetection

# Define module-level Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PlankDetection(ExerciseDetection):
    PREDICTION_PROBABILITY_THRESHOLD = 0.6

    def __init__(self, app):
        super().__init__(app, "plank/model", "plank_model.pkl", "plank_input_scaler.pkl")
        self.stages = {
            "C": "correct",
            "L": "low back",
            "H": "high back"
        }
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False

    def detect(self, mp_results, image, timestamp, stages: dict):
        try:
            row = self.extract_important_keypoints(mp_results, self.important_landmarks)
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
                if self.previous_stage != current_stage:
                    self.results.append({"stage": current_stage, "frame": image.copy(), "timestamp": timestamp})
                    self.has_error = True
            else:
                self.has_error = False

            self.previous_stage = current_stage

            landmark_color, connection_color = self.get_drawing_color(self.has_error)
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
            print(f"Error while detecting plank errors: {e}")
            raise