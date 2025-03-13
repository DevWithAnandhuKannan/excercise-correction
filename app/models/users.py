from flask_login import UserMixin
from app import mongo, login_manager
from flask_bcrypt import generate_password_hash, check_password_hash

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": user_id})
    return User(user_data) if user_data else None

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.email = user_data["email"]
        self.password_hash = user_data["password"]

    @staticmethod
    def get_by_email(email):
        user_data = mongo.db.users.find_one({"email": email})
        return User(user_data) if user_data else None

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def create_user(name, email, age, gender, phone, password):
        hashed_pw = generate_password_hash(password).decode('utf-8')
        mongo.db.users.insert_one({
            "name": name,
            "email": email,
            "age": age,
            "gender": gender,
            "phone": phone,
            "password": hashed_pw
        })
