from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_socketio import SocketIO, emit
from flask_login import login_user, logout_user, current_user
from app.models.users import User
from app.views.forms import LoginForm, RegisterForm, ResetPasswordForm
from flask_bcrypt import generate_password_hash
from bson import ObjectId
import base64
import cv2
from app.core.plank.plank_detection import PlankDetection
from app.core.squat.squat_detection import SquatDetection
from app.core.bicep.bicep_detection import BicepDetection
from app import mongo, app

# Initialize Flask-SocketIO with the app from app/__init__.py
socketio = SocketIO(app, cors_allowed_origins="*")

auth_bp = Blueprint("auth", __name__)

# Initialize exercise detectors
detectors = {}
streaming_flags = {}  # Track streaming state per exercise type

def init_app_resources(app):
    global detectors
    detectors["plank"] = PlankDetection(app)
    detectors["squat"] = SquatDetection(app)
    detectors["bicep"] = BicepDetection(app)
    app.teardown_appcontext(cleanup)

# WebSocket handler for starting exercise stream
@socketio.on("start_exercise")
def handle_exercise(data):
    exercise_type = data.get("exercise_type")
    detector = detectors.get(exercise_type)
    
    if not detector:
        emit("error", {"message": "Invalid exercise type"})
        return

    if streaming_flags.get(exercise_type, False):
        emit("error", {"message": f"{exercise_type.capitalize()} stream already running"})
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit("error", {"message": "Camera not available"})
        return

    streaming_flags[exercise_type] = True
    print(f"Starting exercise stream for: {exercise_type}")  # Debug
    frame_count = 0
    try:
        while streaming_flags.get(exercise_type, False):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")  # Debug
                emit("error", {"message": "Failed to capture frame"})
                break
            
            processed_frame = detector.process_frame(frame, detector.stages)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            emit("exercise_frame", {"image": frame_base64}, broadcast=True)
            frame_count += 1
            print(f"Frame {frame_count} sent for {exercise_type}")  # Debug
    except Exception as e:
        print(f"Error in handle_exercise: {e}")  # Debug
        emit("error", {"message": str(e)})
    finally:
        cap.release()
        streaming_flags[exercise_type] = False
        print(f"Camera released for {exercise_type}, total frames: {frame_count}")  # Debug

# WebSocket handler for stopping exercise stream
@socketio.on("stop_exercise")
def stop_exercise(data):
    exercise_type = data.get("exercise_type")
    if exercise_type in streaming_flags and streaming_flags[exercise_type]:
        streaming_flags[exercise_type] = False
        emit("exercise_stopped", {"message": f"{exercise_type.capitalize()} stream stopped"})
        print(f"Stopped exercise stream for: {exercise_type}")  # Debug
    else:
        emit("error", {"message": f"No active {exercise_type} stream to stop"})

# Routes (unchanged from your last corrected version)
@auth_bp.route('/')
def index():
    user_data = None
    if "user_id" in session:
        try:
            user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
        except Exception as e:
            print(f"Error fetching user data: {e}")
    return render_template("index.html", user=user_data)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        user_data = {
            "name": form.name.data,
            "email": form.email.data,
            "age": form.age.data,
            "gender": form.gender.data,
            "phone": form.phone.data,
            "password": form.password.data
        }
        User.create_user(**user_data)
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("auth.login"))
    return render_template("register.html", form=form)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get_by_email(form.email.data)
        if user and user.check_password(form.password.data):
            login_user(user)
            session["user_id"] = str(user.id)
            print(f"User logged in: {user.id}, Session user_id: {session['user_id']}")
            print(f"Current user authenticated: {current_user.is_authenticated}")
            flash("Login successful!", "success")
            next_page = request.args.get("next", url_for("auth.dashboard"))
            print(f"Redirecting to: {next_page}")
            return redirect(next_page)
        flash("Invalid email or password", "danger")
    return render_template("login.html", form=form)

@auth_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access your dashboard.", "warning")
        return redirect(url_for("auth.login"))  # Redirect instead of rendering login

    try:
        user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
        if not user_data:
            flash("User not found.", "danger")
            return redirect(url_for("auth.login"))

        print(f"Dashboard accessed, Session user_id: {session['user_id']}")
        return render_template("dashboard.html", user=user_data)
    except Exception as e:
        print(f"Error fetching user data: {e}")
        flash("An error occurred while fetching your data.", "danger")
        return redirect(url_for("auth.login"))

@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))

@auth_bp.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = mongo.db.users.find_one({"email": form.email.data})  # Fetch user from MongoDB
        if user:
            hashed_pw = generate_password_hash(form.password.data)  # ✅ No need to decode
            mongo.db.users.update_one({"_id": ObjectId(user["_id"])}, {"$set": {"password": hashed_pw}})
            flash("Password reset successfully. Please login.", "success")
            return redirect(url_for("auth.login"))
        flash("Email not found!", "danger")
    return render_template("reset_password.html", form=form)


@auth_bp.route("/exercise")
def exercise():
    user_data = None
    if "user_id" in session:
        try:
            user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
            return render_template("exercise.html", user=user_data)
        except Exception as e:
            print(f"Error fetching user data: {e}")
            flash("An error occurred while fetching your data.", "danger")
            return redirect(url_for("auth.login"))

    return redirect(url_for("auth.login"))

@auth_bp.route("/user/plank")
def plank():
    user_data = None
    if "user_id" in session:
        try:
            user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
        except Exception as e:
            print(f"Error fetching user data: {e}")
            flash("An error occurred while fetching your data.", "danger")
    return render_template("exercise/plank.html", user=user_data)

@auth_bp.route("/user/squat")
def squat():
    user_data = None
    if "user_id" in session:
        try:
            user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
        except Exception as e:
            print(f"Error fetching user data: {e}")
            flash("An error occurred while fetching your data.", "danger")
    return render_template("exercise/squat.html", user=user_data)

@auth_bp.route("/user/bicep")
def bicep():
    user_data = None
    if "user_id" in session:
        try:
            user_data = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
        except Exception as e:
            print(f"Error fetching user data: {e}")
            flash("An error occurred while fetching your data.", "danger")
    return render_template("exercise/bicep.html", user=user_data)

def cleanup(exception=None):
    print("Cleanup called during app shutdown, skipping Pose closure")  # Debug