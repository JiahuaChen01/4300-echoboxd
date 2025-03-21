import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import similarity as sim
import pandas as pd
import re

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
def json_search(query1, query2, query3):
    # Combine and tokenize user input
    combined_query = f"{query1} {query2} {query3}"
    query_tokens = sim.tokenize(combined_query)

    # Tokenize all reviews if not already done
    reviews_df['toks'] = reviews_df['review'].apply(sim.tokenize)

    # Score each review based on how many query tokens it contains
    def score_row(tokens):
        return sum(1 for token in query_tokens if token in tokens)

    reviews_df['match_score'] = reviews_df['toks'].apply(score_row)

    # Sort by score and get top 10
    top_reviews = reviews_df.sort_values(by='match_score', ascending=False).head(10).copy()

    # Format and rename for frontend
    if 'genre' not in top_reviews.columns:
        top_reviews['genre'] = "Unknown"
    top_reviews.rename(columns={'review': 'description', 'rating': 'imdb_rating'}, inplace=True)
    top_reviews["title"] = top_reviews["title"].apply(lambda x: re.sub(r"\([0-9]{4}\)", "", x))

    matches_filtered = top_reviews[['title', 'year', 'genre', 'description', 'imdb_rating']]
    return matches_filtered.to_json(orient='records', force_ascii=False)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/movies")
def movie_search():
    text1, text2, text3 = [request.args.get(
        title) for title in ("title1", "title2", "title3")]
    return json_search(text1, text2, text3)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
