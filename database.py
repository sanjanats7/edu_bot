from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# MongoDB connection setup
client = MongoClient(
    "mongodb+srv://surabhix16:db_password@cluster0.idk98.mongodb.net/?tls=true"
)  # Replace with your MongoDB URI
db = client["edu_bot"]  # Database name
users_col = db["users"]  # Collection for user data
queries_col = db["queries"]  # Collection for storing queries and responses


# User registration
def register_user(username, password):
    if users_col.find_one({"username": username}):
        return "User already exists"
    hashed_pw = generate_password_hash(password)
    users_col.insert_one(
        {"username": username, "password": hashed_pw, "documents": [], "queries": []}
    )
    return "User registered successfully"


# User login
def login_user(username, password):
    user = users_col.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return True
    return False


# Save query and response
def save_query(username, query, response):
    users_col.update_one(
        {"username": username},
        {"$push": {"queries": {"query": query, "response": response}}},
    )


# Fetch user chat history
def get_user_history(username):
    user = users_col.find_one({"username": username})
    return user.get("queries", []) if user else []
