import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import similarity as sim
from stats import get_user_similarities
import pandas as pd
import re
from tfidf_utils import compute_tf, compute_idf, compute_tf_idf, cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

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

with open("metacritic-reviews.json", "r", encoding="utf-8") as file1:
    metacritic_summaries = json.load(file1)
    metacritic_df = pd.DataFrame(metacritic_summaries)
    metacritic_df.rename(columns={'summary': 'review', 'User rating': 'rating', 'Movie name': 'title', 'Release Date': 'year'}, inplace=True)
    metacritic_df.drop(columns={'Rating', 'Website rating'})
    reviews_df = pd.concat([reviews_df, metacritic_df], ignore_index=True)

with open("tmdb_5000_movies.json", "r", encoding="utf-8") as file2:
    tmdb_overviews = json.load(file2)
    tmdb_df = pd.DataFrame(tmdb_overviews)
    tmdb_df.rename(columns={'overview': 'review', 'release_date': 'year', 'vote_average': 'rating'}, inplace=True)
    tmdb_df.drop(columns={'keywords', 'original_title', 'tagline'})
    reviews_df = pd.concat([reviews_df, tmdb_df], ignore_index=True)

app = Flask(__name__)
CORS(app)

valid_df = reviews_df.dropna(subset=["review"]).copy()
valid_df['toks'] = valid_df['review'].apply(sim.tokenize)
docs = valid_df['toks'].tolist()
idf = compute_idf(docs)
doc_vecs = [compute_tf_idf(compute_tf(doc), idf) for doc in docs]
review_to_index = dict(enumerate(valid_df["review"]))
index_to_review = {i:t for t,i in review_to_index.items()}

movie_to_index = dict(enumerate(valid_df["title"]))
index_to_movie = {i:t for t,i in movie_to_index.items()}

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7,
                            min_df = 2)
td_matrix = vectorizer.fit_transform(valid_df["review"])
docs_compressed, s, words_compressed = svds(td_matrix, k=200)
words_compressed = words_compressed.transpose()

def json_search(query1="", query2="", query3="", query4="", query5=""):
    combined_query = f"{query1} {query2} {query3} {query4} {query5}".strip()
    query_tokens = sim.tokenize(combined_query)
    updated_tokens = []
    for q in query_tokens:
        for toks in valid_df["toks"]:
            min_dist = float("inf")
            min_tok = None
            for tok in toks:
                dist = sim.edit_distance(q, tok)
                if dist < min_dist:
                    min_dist = dist
                    min_tok = tok
            if min_tok == q:
                updated_tokens.append(q)
            else:
                updated_tokens.append(min_tok)
    
    

    
    query_vec = compute_tf_idf(compute_tf(query_tokens), idf)

    

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    from sklearn.preprocessing import normalize
    words_compressed_normed = normalize(words_compressed, axis = 1)

    docs_compressed_normed = normalize(docs_compressed)

    def doc_scores_with_query(words_in, k = 5):
        sims = np.zeros((docs_compressed_normed.shape[0], words_compressed_normed.shape[1]))
        for word in words_in:
            if word not in word_to_index:
                continue
            # fix this later
            if np.sum(sims) == 0:
                sims = docs_compressed_normed.dot(words_compressed_normed[word_to_index[word],:])
            else:
                sims += docs_compressed_normed.dot(words_compressed_normed[word_to_index[word],:])
        return sims

    scores = doc_scores_with_query(query_tokens)
    valid_df["score"] = scores

    top = valid_df.sort_values('score', ascending=False).head(10).copy()

    if 'genre' not in top.columns:
        top['genre'] = "Unknown"
    top.rename(columns={'review': 'description', 'rating': 'imdb_rating'}, inplace=True)
    top['title'] = top['title'].apply(lambda x: re.sub(r"\([0-9]{4}\)", "", x))

    top.drop_duplicates(subset='title', keep='first')
    user_scores = get_user_similarities([query1, query2, query3, query4, query5], top['description'].tolist())
    top['user_scores'] = user_scores

    return top[['title', 'year', 'genre', 'description', 'imdb_rating', 'user_scores']].to_json(orient='records', force_ascii=False)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/movies")
def movie_search():
    titles = [request.args.get(f"title{i}", "") for i in range(1, 6)]
    return json_search(*titles)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
