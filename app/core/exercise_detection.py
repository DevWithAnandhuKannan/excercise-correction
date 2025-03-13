import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ExerciseDetection:
    PREDICTION_PROBABILITY_THRESHOLD = 0.6

    def __init__(self, app, model_dir: str, model_file: str, scaler_file: str):
        self.app = app
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ML_MODEL_PATH = os.path.join(base_dir, model_dir, model_file)
        self.INPUT_SCALER_PATH = os.path.join(base_dir, model_dir, scaler_file)
        self.model = None
        self.input_scaler = None
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False
        self.init_important_landmarks()
        self.load_machine_learning_model()

    def get_static_file_url(self, file_name: str) -> str:
        path = os.path.join(self.app.static_folder, file_name)
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}")
            return None
        return path

    @staticmethod
    def extract_important_keypoints(results, important_landmarks: list) -> list:
        landmarks = results.pose_landmarks.landmark
        data = []
        for lm in important_landmarks:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
        return np.array(data).flatten().tolist()

    @staticmethod
    def get_drawing_color(error: bool) -> tuple:
        LIGHT_BLUE = (244, 117, 66)
        LIGHT_PINK = (245, 66, 230)
        LIGHT_RED = (29, 62, 199)
        LIGHT_YELLOW = (1, 143, 241)
        return (LIGHT_YELLOW, LIGHT_RED) if error else (LIGHT_BLUE, LIGHT_PINK)

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
        if not os.path.exists(self.ML_MODEL_PATH) or not os.path.exists(self.INPUT_SCALER_PATH):
            print(f"Warning: Model files not found at {self.ML_MODEL_PATH} or {self.INPUT_SCALER_PATH}. Detection disabled.")
            return
        try:
            with open(self.ML_MODEL_PATH, "rb") as f:
                self.model = joblib.load(f)
            with open(self.INPUT_SCALER_PATH, "rb") as f2:
                self.input_scaler = joblib.load(f2)
            print(f"Loaded model from: {self.ML_MODEL_PATH}")
            print(f"Loaded scaler from: {self.INPUT_SCALER_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def handle_detected_results(self, video_name: str) -> tuple:
        file_name, _ = video_name.split(".")
        save_folder = self.get_static_file_url("images")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for index, error in enumerate(self.results):
            try:
                image_name = f"{file_name}_{index}.jpg"
                cv2.imwrite(f"{save_folder}/{image_name}", error["frame"])
                self.results[index]["frame"] = f"images/{image_name}"
            except Exception as e:
                print(f"ERROR cannot save frame: {e}")
                self.results[index]["frame"] = None
        return self.results, self.previous_stage

    def clear_results(self) -> None:
        self.previous_stage = "unknown"
        self.results = []
        self.has_error = False

    def detect(self, mp_results, image, timestamp, stages: dict) -> None:
        if not self.model or not self.input_scaler:
            cv2.putText(image, "Model not loaded", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return
        try:
            row = self.extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))
            predicted_class = self.model.predict(X)[0]
            prediction_probability = self.model.predict_proba(X)[0]

            current_stage = stages.get(predicted_class, "unknown")
            if prediction_probability[prediction_probability.argmax()] < self.PREDICTION_PROBABILITY_THRESHOLD:
                current_stage = "unknown"

            if current_stage in stages.values() and current_stage != "correct":
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
            print(f"Error while detecting exercise: {e}")
            raise

    def process_frame(self, frame, stages: dict):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)  # Correct: no image_dimensions
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.detect(results, frame_bgr, timestamp=0, stages=stages)
        return frame_bgr