from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# MongoDB connection setup
client = MongoClient(
    "mongodb+srv://surabhix16:db_password@cluster0.idk98.mongodb.net/?tls=true"
)  # Replace with your MongoDB URI
db = client["edu_bot"]  # Database name
users_col = db["users"]  # Collection for user data


# User registration
def register_user(username, password):
    if users_col.find_one({"username": username}):
        return "User already exists"
    hashed_pw = generate_password_hash(password)
    users_col.insert_one(
        {
            "username": username,
            "password": hashed_pw,
            "documents": [],  # List to store uploaded document names
            "queries": [],  # List to store query-response pairs
        }
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
    if not response:
        raise ValueError("Cannot save an empty response.")
        return
    users_col.update_one(
        {"username": username},
        {
            "$push": {
                "queries": {
                    "question": query,
                    "answer": response,
                }
            }
        },
    )


# Save uploaded document
def save_uploaded_document(username, document_name):
    users_col.update_one(
        {"username": username},
        {"$push": {"documents": document_name}}
    )


# Fetch user chat history
def get_user_history(username):
    # Fetch user from the database
    user = users_col.find_one({"username": username})
    
    # Return an empty list if no user is found
    if not user:
        return []

    # Ensure that each query has 'question' and 'answer' keys
    queries = user.get("queries", [])
    
    # Normalize the queries to ensure each query has the correct format
    history = []
    for query in queries:
        # Default to empty strings if keys are missing
        question = query.get("question", "No question found")
        answer = query.get("answer", "No answer found")
        history.append({"question": question, "answer": answer})
    
    return history