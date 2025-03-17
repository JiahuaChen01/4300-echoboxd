import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open("letterboxd_reviews.json", "r", encoding="utf-8") as file:
    movie_reviews = json.load(file)
    reviews_df = pd.DataFrame(movie_reviews)

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = reviews_df[
        reviews_df['title'].str.lower().str.contains(query.lower()) |
        reviews_df['review'].str.lower().str.contains(query.lower())
    ].copy()  # Explicitly create a copy to avoid warnings

    # Ensure all required fields exist and handle missing values
    if 'genre' not in matches.columns:
        matches['genre'] = "Unknown"  # Set default genre if missing
    
    if 'review' in matches.columns:
        matches.rename(columns={'review': 'description'}, inplace=True)
    else:
        matches['description'] = "No description available."

    if 'rating' in matches.columns:
        matches.rename(columns={'rating': 'imdb_rating'}, inplace=True)
    else:
        matches['imdb_rating'] = "N/A"
    
    matches_filtered = matches[['title', 'year', 'genre', 'description', 'imdb_rating']]
    
    return matches_filtered.to_json(orient='records', force_ascii=False)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/movies")
def movie_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)