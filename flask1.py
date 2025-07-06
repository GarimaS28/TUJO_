from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import random
import jwt  # For token-based authentication

app = Flask(__name__)
CORS(app)

# -------------------------------
# MongoDB Connection Setup
# -------------------------------
client = MongoClient("mongodb://127.0.0.1:27017")
db = client["diaryAppDB"]
users_collection = db["users"]
entries_collection = db["entries"]

# Secret key for JWT token (should be stored securely in production)
app.config['SECRET_KEY'] = 'your_secret_key'

# -------------------------------
# Load the Trained Mood Classifier
# -------------------------------
class MoodClassifier(nn.Module):
    def __init__(self, num_moods):
        super(MoodClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, num_moods)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

# Set the number of moods as per your training (we assume 8 moods)
num_moods = 8
mood_classifier = MoodClassifier(num_moods)
# Load your trained mood detection model
mood_classifier.load_state_dict(torch.load("mood_classifier.pth", map_location=torch.device('cpu')))
mood_classifier.eval()

# Load the tokenizer that matches your training
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a complete mood mapping (ensure this matches your training order)
mood_mapping = {
    0: "Sadness",
    1: "Happiness",
    2: "Anger",
    3: "Neutral",
    4: "Love",
    5: "Joy",
    6: "Fear",
    7: "Surprise"
}

# -------------------------------
# Real-Time Recommendation Function
# -------------------------------
def generate_recommendation(mood, area_of_interest, previous_entries):
    """
    Generates a personalized recommendation based on:
    - The detected mood
    - The user's area of interest
    - (Optional) Their previous diary entries (to adapt the message)
    """
    # Baseline recommendations for each mood:
    baseline = {
        "Sadness": "Maybe try listening to your favorite calming music or watch a comforting movie.",
        "Happiness": "Keep celebrating your happiness—sharing that joy with a friend might be great!",
        "Anger": "Take a deep breath, perhaps go for a short walk, or try a few relaxation techniques.",
        "Neutral": "You might enjoy exploring a new hobby or activity to spark some excitement.",
        "Love": "Consider expressing your feelings or spending quality time with someone special.",
        "Joy": "Your energy is contagious! Perhaps channel it into something creative.",
        "Fear": "Take a moment to reflect and consider talking to someone you trust about your worries.",
        "Surprise": "Embrace the unexpected—maybe try something new or spontaneous today."
    }
    # Add a history note if similar mood entries exist
    history_note = ""
    if previous_entries:
        history_note = " Also, based on your past experiences when you felt this way, you might consider similar activities."
    
    recommendation_text = baseline.get(mood, "Take a moment for yourself.")
    # Optionally, if the user's area of interest isn't mentioned, add it to the message.
    if area_of_interest and area_of_interest.lower() not in recommendation_text.lower():
        recommendation_text += f" Since you enjoy {area_of_interest}, perhaps you can incorporate that too."
    
    return recommendation_text + history_note

# -------------------------------
# Token Required Decorator
# -------------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token is missing!"}), 403
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['email']  # Assuming email is stored in the token
        except:
            return jsonify({"error": "Token is invalid!"}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# -------------------------------
# Recommendation Endpoint
# -------------------------------
@app.route("/get_personalized_recommendation", methods=["POST"])
@token_required
def get_personalized_recommendation(current_user):
    """
    Expects a JSON payload with:
      - "entry": the diary entry text
    """
    data = request.get_json()
    diary_entry = data.get("entry", "")
    
    if not diary_entry:
        return jsonify({"error": "Diary entry is required"}), 400

    # Retrieve user details from MongoDB
    user = users_collection.find_one({"email": current_user})
    if not user:
        return jsonify({"error": "User not found"}), 404
    area_of_interest = user.get("areaOfInterest", "general")

    # Use the mood classifier to predict the mood from the diary entry
    inputs = tokenizer(diary_entry, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = mood_classifier(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        predicted_idx = torch.argmax(outputs, dim=1).item()
    detected_mood = mood_mapping.get(predicted_idx, "Neutral")
    
    # Optionally, retrieve previous diary entries by this user with the same mood
    previous_entries = list(entries_collection.find({"user_email": current_user, "mood": detected_mood}))
    
    # Generate a personalized recommendation
    recommendation = generate_recommendation(detected_mood, area_of_interest, previous_entries)
    personalized_message = f"It's okay to feel {detected_mood.lower()}. {recommendation}"
    
    return jsonify({
        "mood": detected_mood,
        "recommendation": recommendation,
        "personalized_message": personalized_message
    })

# -------------------------------
# Fetch Entries Endpoint
# -------------------------------
@app.route("/fetch-entry", methods=["GET"])
@token_required
def fetch_entry(current_user):
    """
    Fetches diary entries for the logged-in user for a specific date.
    Expects a query parameter:
      - "date": the date in YYYY-MM-DD format
    """
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "Date is required"}), 400

    # Fetch entries for the logged-in user for the specified date
    entries = list(entries_collection.find({"user_email": current_user, "date": date}))
    return jsonify({"entries": entries})

# -------------------------------
# Add Entry Endpoint
# -------------------------------
@app.route("/add_entry", methods=["POST"])
@token_required
def add_entry(current_user):
    """
    Adds a new diary entry for the logged-in user.
    Expects a JSON payload with:
      - "entry": the diary entry text
    """
    data = request.get_json()
    diary_entry = data.get("entry", "")
    
    if not diary_entry:
        return jsonify({"error": "Diary entry is required"}), 400

    # Use the mood classifier to predict the mood from the diary entry
    inputs = tokenizer(diary_entry, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = mood_classifier(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        predicted_idx = torch.argmax(outputs, dim=1).item()
    detected_mood = mood_mapping.get(predicted_idx, "Neutral")

    # Save the entry to MongoDB
    entry_data = {
        "user_email": current_user,
        "entry": diary_entry,
        "mood": detected_mood,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    entries_collection.insert_one(entry_data)
    
    return jsonify({
        "message": "Diary entry saved successfully!",
        "mood": detected_mood
    })

# -------------------------------
# Get Weekly Moods Endpoint
# -------------------------------
@app.route("/get_weekly_moods", methods=["GET"])
@token_required
def get_weekly_moods(current_user):
    """
    Fetches mood data for the logged-in user for the past week.
    """
    # Calculate the date range for the past week
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Fetch entries for the logged-in user within the date range
    entries = list(entries_collection.find({
        "user_email": current_user,
        "date": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")}
    }))
    
    # Prepare mood data for the chart
    mood_data = []
    for entry in entries:
        mood_data.append({
            "date": entry["date"],
            "mood_index": list(mood_mapping.keys())[list(mood_mapping.values()).index(entry["mood"])]
        })
    
    return jsonify(mood_data)

# -------------------------------
# Run the Flask App on Port 5001
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)