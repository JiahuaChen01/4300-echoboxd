from tfidf_utils import compute_tf, compute_idf, compute_tf_idf, cosine_similarity
import similarity as sim
import numpy as np

def get_user_similarities(query_inputs, movie_reviews):
    # Filter out empty user queries
    clean_inputs = [q.strip() for q in query_inputs if q.strip()]
    if not clean_inputs:
        return {}

    user_tokens = [sim.tokenize(query) for query in clean_inputs]
    doc_tokens = [sim.tokenize(str(review)) for review in movie_reviews]

    idf = compute_idf(doc_tokens + user_tokens)

    doc_vecs = [compute_tf_idf(compute_tf(doc), idf) for doc in doc_tokens]
    user_vecs = [compute_tf_idf(compute_tf(user), idf) for user in user_tokens]

    # Compute cosine similarities for each user vs each movie
    user_scores = []
    for doc_vec in doc_vecs:
        scores = [cosine_similarity(doc_vec, user_vec) for user_vec in user_vecs]
        user_scores.append(scores)

    return user_scores
