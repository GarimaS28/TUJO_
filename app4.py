import pandas as pd
import random
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from datetime import datetime, timedelta
from training3 import detect_mood  # Ensure this imports your mood detection function
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
db = client["diaryAppDB"]
entries_collection = db["entries"]
@app.route("/get_recommendation", methods=["POST"])
def get_recommendation():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form
    
    entry_text = data.get("entry", "")
    if not entry_text:
        return jsonify({"error": "Entry cannot be empty"}), 400
    
    # Detect mood using your existing function
    mood = detect_mood(entry_text)
    
    # Make sure mood is lowercase for consistent matching
    mood = mood.lower()
    
    # Define fallback recommendations if file isn't found
    fallback_recommendations = {
        "sadness": ["Listen to calming music", "Take a walk in nature", "Call a friend"],
        "happiness": ["Celebrate with friends", "Practice gratitude", "Dance freely"],
        "anger": ["Try deep breathing exercises", "Go for a run", "Listen to relaxing music"],
        "neutral": ["Explore a new hobby", "Read an interesting book", "Go for a walk"],
        "love": ["Write a heartfelt message", "Spend time with loved ones", "Express gratitude"],
        "joy": ["Capture this moment in a journal", "Enjoy your favorite treat", "Share your happiness"],
        "fear": ["Talk to someone about your fear", "Practice deep breathing", "Write down your thoughts"],
        "surprise": ["Embrace the unexpected moment", "Share the surprise with a friend", "Take a deep breath"]
    }
    
    try:
        # Try to load the CSV file with absolute path
        import os
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendations.csv")
        
        if os.path.exists(file_path):
            recommendations_df = pd.read_csv(file_path)
            
            # Convert column and mood to lowercase for case-insensitive matching
            recommendations_df["Detected Mood"] = recommendations_df["Detected Mood"].str.lower()
            
            # Filter recommendations by detected mood
            mood_recommendations = recommendations_df[recommendations_df["Detected Mood"] == mood]
            
            if not mood_recommendations.empty:
                # Randomly select one recommendation
                recommendation = random.choice(mood_recommendations["Suggested Activity"].tolist())
            else:
                # Use fallback if mood not found in dataset
                recommendation = random.choice(fallback_recommendations.get(mood, ["Take some time for yourself today."]))
        else:
            # File doesn't exist, use fallback
            recommendation = random.choice(fallback_recommendations.get(mood, ["Take some time for yourself today."]))
    except Exception as e:
        print(f"Error fetching recommendation: {e}")
        # Use fallback recommendations if file can't be loaded
        recommendation = random.choice(fallback_recommendations.get(mood, ["Take some time for yourself today."]))
    
    return jsonify({"mood": mood, "recommendation": recommendation})
@app.route("/add_entry", methods=["POST"])
def add_entry():
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        entry_text = data.get("entry", "")
        user_email = data.get("user_email", "")
        
        if not entry_text or not user_email:
            return jsonify({"error": "Entry and user email cannot be empty"}), 400

        # Detect mood
        mood = detect_mood(entry_text)

        # Get user-specific collection
        user_collection_name = f"entries_{user_email.replace('@', '_').replace('.', '_')}"
        user_collection = db[user_collection_name]

        # Save entry
        entry_data = {
            "date": datetime.now().date().isoformat(),
            "entry": entry_text,
            "mood": mood,
            "timestamp": datetime.now()
        }
        result = user_collection.insert_one(entry_data)
        
        return jsonify({
            "message": "Entry saved!", 
            "mood": mood, 
            "id": str(result.inserted_id)
        })

@app.route("/fetch-entry", methods=["GET"])
def fetch_entry():
    date_str = request.args.get("date")
    user_email = request.args.get("user_email")
    
    if not date_str or not user_email:
        return jsonify({"error": "Date and user email are required"}), 400

    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        # Get user-specific collection
        user_collection_name = f"entries_{user_email.replace('@', '_').replace('.', '_')}"
        user_collection = db[user_collection_name]
        
        # Find entries for the specified date
        entries = list(user_collection.find({"date": date_str}))
        
        # Remove MongoDB _id field
        for entry in entries:
            entry.pop('_id', None)
            
        return jsonify({"entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_weekly_moods", methods=["GET"])
def get_weekly_moods():
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    # Update mood mapping to match your training model
    mood_to_index = {
        "sadness": 0,
        "happiness": 1,
        "anger": 2,
        "neutral": 3,
        "love": 4,
        "joy": 5,
        "fear": 6,
        "surprise": 7
    }

    today = datetime.now().date()
    week_ago = today - timedelta(days=6)

    try:
        user_collection_name = f"entries_{user_email.replace('@', '_').replace('.', '_')}"
        user_collection = db[user_collection_name]

        entries = list(user_collection.find({
            "date": {
                "$gte": week_ago.isoformat(),
                "$lte": today.isoformat()
            }
        }))

        # Prepare daily mood data
        mood_data = {}
        for entry in entries:
            date_val = entry.get("date")
            mood_str = entry.get("mood", "").lower()
            
            if date_val and mood_str in mood_to_index:
                if date_val not in mood_data:
                    mood_data[date_val] = {
                        'sum': mood_to_index[mood_str],
                        'count': 1
                    }
                else:
                    mood_data[date_val]['sum'] += mood_to_index[mood_str]
                    mood_data[date_val]['count'] += 1

        # Prepare result for all 7 days
        result = []
        for i in range(7):
            current_date = (week_ago + timedelta(days=i)).isoformat()
            if current_date in mood_data:
                avg_index = mood_data[current_date]['sum'] / mood_data[current_date]['count']
                result.append({
                    "date": current_date,
                    "mood_index": round(avg_index)  # Round to nearest integer
                })
            else:
                # Default to neutral if no entries
                result.append({
                    "date": current_date,
                    "mood_index": 3  # Neutral
                })

        return jsonify(result)

    except Exception as e:
        print(f"Error in get_weekly_moods: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
