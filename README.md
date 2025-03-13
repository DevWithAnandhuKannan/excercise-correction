Below is a more structured and detailed `README.md` for your **Exercise Correction App**, designed to give readers a clear understanding of the project’s purpose, structure, and functionality at a glance. It organizes the content into well-defined sections, improves readability with consistent formatting, and integrates the directory structure and algorithms seamlessly.

---

# Exercise Correction App

![Fitness Image](static/images/fitness.jpg)

The **Exercise Correction App** is an AI-powered fitness tool that provides real-time feedback on exercise form. Using computer vision and machine learning, it analyzes movements during planks, squats, and bicep curls, helping users improve technique and avoid injuries. Built with Flask, MongoDB, and MediaPipe, it offers a user-friendly interface with live video streaming and secure authentication.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Algorithms Used](#algorithms-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Training Models](#training-models)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Overview
This application harnesses AI to enhance workout experiences by detecting and correcting exercise form in real time. It uses MediaPipe for pose estimation and pre-trained machine learning models to classify exercise stages and identify errors, such as sagging during planks or leaning too far back during bicep curls. The app integrates a Flask backend with WebSocket streaming for live video feedback and MongoDB for user data management.

---

## Features
- **Real-Time Analysis**: Detects pose landmarks and provides instant feedback via webcam.
- **Exercise-Specific Feedback**: Corrects form for planks, squats, and bicep curls with detailed error messages.
- **User Authentication**: Secure login, registration, and password reset with a glassmorphism UI.
- **Live Streaming**: Streams video feed with overlaid analysis using Socket.IO.
- **Data Persistence**: Stores user profiles and passwords (hashed) in MongoDB.

---

## Project Structure
The app is organized into modular directories for clarity and maintainability. Below is the structure with descriptions:

```
app/
├── __init__.py                    # Initializes the Flask app and configurations
├── __pycache__/                   # Compiled Python bytecode (auto-generated)
│   └── __init__.cpython-39.pyc
├── controller/                    # Handles routes and logic
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-39.pyc
│   │   └── user_controller.cpython-39.pyc
│   └── user_controller.py         # Defines routes (e.g., /login, /user/plank) and WebSocket handlers
├── core/                          # Core exercise detection modules
│   ├── __pycache__/
│   │   └── exercise_detection.cpython-39.pyc
│   ├── bicep/                    # Bicep curl detection module
│   │   ├── __pycache__/
│   │   │   └── bicep_detection.cpython-39.pyc
│   │   ├── app.py                # Standalone bicep app (optional)
│   │   ├── bicep_detection.py    # Bicep detection logic (Logistic Regression)
│   │   ├── model/                # Pre-trained model files
│   │   │   ├── bicep_curl_input_scaler.pkl  # Feature scaler
│   │   │   └── bicep_curl_model.pkl         # Trained model
│   │   ├── test.csv              # Test dataset
│   │   ├── train.csv             # Training dataset
│   │   └── train_bicep_model.ipynb  # Jupyter notebook for model training
│   ├── exercise_detection.py     # Base class for all exercise detection
│   ├── plank/                    # Plank detection module
│   │   ├── __pycache__/
│   │   │   └── plank_detection.cpython-39.pyc
│   │   ├── app.py
│   │   ├── model/
│   │   │   ├── plank_input_scaler.pkl
│   │   │   └── plank_model.pkl
│   │   ├── plank_detection.py    # Plank detection logic (Random Forest)
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── train_plank_model.ipynb
│   └── squat/                    # Squat detection module
│       ├── __pycache__/
│       │   └── squat_detection.cpython-39.pyc
│       ├── app.py
│       ├── model/
│       │   ├── scaler.pkl
│       │   └── squat_model.pkl
│       ├── squat_detection.py    # Squat detection logic (SVM)
│       ├── test.csv
│       ├── train.csv
│       └── train_squat_model.ipynb
├── models/                       # Database models
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-39.pyc
│   │   └── users.cpython-39.pyc
│   └── users.py                 # User model for MongoDB operations
├── requirements.txt              # List of Python dependencies
├── static/                       # Static assets (CSS, JS, images)
│   ├── css/
│   │   └── style.css            # Custom styles
│   ├── images/
│   │   └── fitness.jpg          # Sample fitness image
│   └── js/
│       └── script.js            # Client-side JavaScript (e.g., Socket.IO)
├── templates/                    # HTML templates
│   ├── base.html                # Base template with Bootstrap
│   ├── components/
│   │   └── header.html          # Reusable header component
│   ├── dashboard.html           # User dashboard
│   ├── exercise/                # Exercise-specific pages
│   │   ├── bicep.html
│   │   ├── plank.html
│   │   └── squat.html
│   ├── exercise.html            # Exercise selection page
│   ├── index.html              # Landing page
│   ├── login.html              # Login page
│   ├── register.html           # Registration page
│   └── reset_password.html     # Password reset page
└── views/                       # Form definitions
    ├── __init__.py
    ├── __pycache__/
    │   ├── __init__.cpython-39.pyc
    │   └── forms.cpython-39.pyc
    └── forms.py                 # WTForms for login, register, reset password
```

---

## Algorithms Used
Each exercise detection module uses a specific machine learning algorithm trained on pose landmark data from MediaPipe. Here’s a breakdown:

### Plank Detection (`plank_detection.py`)
- **Algorithm**: Random Forest Classifier
- **Purpose**: Classifies plank stages (e.g., "correct", "sag", "pike").
- **Features**: Keypoint coordinates (x, y, z, visibility) from shoulders, hips, and ankles.
- **Training Data**: `train.csv`
- **Scaler**: `plank_input_scaler.pkl` (normalizes features)
- **Model File**: `plank_model.pkl`
- **Training Notebook**: `train_plank_model.ipynb`

### Squat Detection (`squat_detection.py`)
- **Algorithm**: Support Vector Machine (SVM)
- **Purpose**: Predicts squat stages ("up", "down", "middle") and checks placement errors.
- **Features**: Landmarks like nose, shoulders, hips, knees, and ankles.
- **Training Data**: `train.csv`
- **Scaler**: `scaler.pkl`
- **Model File**: `squat_model.pkl`
- **Additional Logic**: Rule-based checks for foot-shoulder and knee-foot ratios.
- **Training Notebook**: `train_squat_model.ipynb`

### Bicep Curl Detection (`bicep_detection.py`)
- **Algorithm**: Logistic Regression
- **Purpose**: Classifies posture ("L" for lean too far back, "C" for correct).
- **Features**: Landmarks including nose, shoulders, elbows, wrists, and hips.
- **Training Data**: `train.csv`
- **Scaler**: `bicep_curl_input_scaler.pkl`
- **Model File**: `bicep_curl_model.pkl`
- **Additional Logic**: Angle-based analysis (`BicepPoseAnalysis`) for errors like loose upper arm or weak peak contraction.
- **Training Notebook**: `train_bicep_model.ipynb`

Models are saved as `.pkl` files and loaded with `joblib`.

---

## Installation
Follow these steps to set up the app locally:

### Prerequisites
- Python 3.9+
- MongoDB (local or cloud)
- Git

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/exercise-correction-app.git
   cd exercise-correction-app
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB**:
   - Install MongoDB (e.g., `brew install mongodb` on macOS) or use MongoDB Atlas.
   - Start MongoDB: `mongod` (if local).
   - Update `app/__init__.py`:
     ```python
     app.config["MONGO_URI"] = "mongodb://localhost:27017/exercise_db"
     ```

5. **Run the App**:
   ```bash
   python runner.py
   ```
   - Open `http://127.0.0.1:5000` in your browser.

---

## Usage
### Accessing the App
- **Landing Page**: `/` - Overview with sign-in/register options.
- **Authentication**:
  - **Register**: `/register` - Create an account.
  - **Login**: `/login` - Sign in to access the dashboard.
  - **Reset Password**: `/reset_password` - Update your password.

### Exercise Detection
1. Navigate to an exercise page:
   - `/user/plank`
   - `/user/squat`
   - `/user/bicep`
2. Click "Start [Exercise]" to begin webcam streaming.
3. Perform the exercise; feedback (e.g., counts, errors) appears on the video.
4. Click "Stop [Exercise]" to end the session.

---

## Dependencies
Key dependencies from `requirements.txt`:
- **Flask**: Web framework
- **pymongo**: MongoDB driver
- **mediapipe**: Pose estimation
- **opencv-python**: Video processing
- **scikit-learn**: Machine learning
- **flask-socketio**: WebSocket support
- **flask-wtf**: Form validation
- **werkzeug**: Security utilities

Install with: `pip install -r requirements.txt`.

---

## Training Models
To retrain or update models:
1. **Prepare Data**:
   - Update `train.csv` and `test.csv` in the respective exercise folder (`bicep/`, `plank/`, `squat/`).
   - Data should include pose landmarks (x, y, z, visibility) and labels.
2. **Run Notebook**:
   - Open the `.ipynb` file (e.g., `train_bicep_model.ipynb`) in Jupyter Notebook.
   - Execute all cells to train the model and scaler.
3. **Save Outputs**:
   - Models are saved as `.pkl` files in the `model/` subdirectories.

---

## Contributing
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open a pull request on GitHub.

---

## License
This project is licensed under the [MIT License](LICENSE). Create a `LICENSE` file if not present.

---

## Contact
For inquiries or feedback:
- **Email**: [connect.anandhukannan@gmail.com](mailto:connect.anandhukannan@gmail.com)
- **GitHub**: [DevWithAnandhuKannan](https://github.com/DevWithAnandhuKannan/)

---

### Notes
- Replace placeholders (e.g., `yourusername`, `your-email@example.com`) with your actual details.
- Add a `runner.py` reference if it’s not in the root (e.g., move it to `/app/` and update the README).
- The structure assumes `runner.py` or similar is outside `/app/`. Adjust paths if needed.

This README provides a clear, structured guide for users and developers. Let me know if you’d like further refinements!