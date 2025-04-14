import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import similarity as sim
import pandas as pd
import re
from tfidf_utils import compute_tf, compute_idf, compute_tf_idf, cosine_similarity


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

with open("metacritic_reviews.json", "r", encoding="utf-8") as file:
    metacritic_summaries = json.load(file)
    metacritic_df = pd.DataFrame(metacritic_summaries)

app = Flask(__name__)
CORS(app)

def json_search(query1, query2, query3):
    combined_query = f"{query1} {query2} {query3}".strip()
    query_tokens = sim.tokenize(combined_query)

    valid_df = reviews_df.dropna(subset=["review"]).copy()
    metacritic_valid_df = metacritic_df.dropna(subset=["summary"]).copy()
    valid_df['toks'] = valid_df['review'].apply(sim.tokenize)
    metacritic_valid_df['toks'] = metacritic_valid_df['summary'].apply(sim.tokenize)

    docs = valid_df['toks'].tolist()
    docs.append(metacritic_valid_df['toks'].tolist())
    idf = compute_idf(docs)
    doc_vecs = [compute_tf_idf(compute_tf(doc), idf) for doc in docs]
    query_vec = compute_tf_idf(compute_tf(query_tokens), idf)

    valid_df['score'] = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]
    top = valid_df.sort_values('score', ascending=False).head(10).copy()

    if 'genre' not in top.columns:
        top['genre'] = "Unknown"
    top.rename(columns={'review': 'description', 'rating': 'imdb_rating'}, inplace=True)
    top.rename(columns={'summary': 'description', 'User rating': 'imdb_rating', 'Movie name': 'title', 'Release Date': 'year'}, inplace=True)
    top['title'] = top['title'].apply(lambda x: re.sub(r"\([0-9]{4}\)", "", x))

    return top[['title', 'year', 'genre', 'description', 'imdb_rating']].to_json(orient='records', force_ascii=False)
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
