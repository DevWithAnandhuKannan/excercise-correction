from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
import os
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure
from flask_cors import CORS
from flask_bcrypt import Bcrypt

load_dotenv()

mongo = PyMongo()
login_manager = LoginManager()
bcrypt = Bcrypt()

app = Flask(__name__)  # Define app globally
CORS(app)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

def create_app():
    if not app.config["SECRET_KEY"]:
        raise ValueError("❌ SECRET_KEY is missing in the environment variables!")
    if not app.config["MONGO_URI"]:
        raise ValueError("❌ MONGO_URI is missing in the environment variables!")
    mongo.init_app(app)
    try:
        mongo.db.command("ping")
        print("✅ Database Connected Successfully!")
    except ConnectionFailure:
        print("❌ Database Connection Failed!")
    login_manager.init_app(app)
    bcrypt.init_app(app)
    login_manager.login_view = "auth.login"
    try:
        from app.controller.user_controller import auth_bp, init_app_resources
        app.register_blueprint(auth_bp)
        init_app_resources(app)
    except ImportError as e:
        print(f"❌ Failed to import blueprint: {e}")
    return app

@login_manager.user_loader
def load_user(user_id):
    from app.models.users import User
    return User.get_by_id(user_id)