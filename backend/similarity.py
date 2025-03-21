import numpy as np
import re

def tokenize(review: str):
    if not isinstance(review, str):
        return []
    return [tok for tok in re.findall(r"[A-Za-z']+", review.lower())]

def build_inverted_index(data: list[dict]) -> dict:
    idx = {}
    i = 0
    for review in data:
      count = {}
      for tok in review["toks"]:
        if tok in count.keys():
          count[tok] += 1
        else:
          count[tok] = 1
      for k in count.keys():
        if k in idx.keys():
          idx[k].append((i,count[k]))
        else:
          idx[k] = [(i,count[k])]
      i+=1
    return idx

#Args: {movie title:[list of reviews]}, [list of common English words]
def create_word_occurrence_matrix(data:  dict[str,list], good_types: list[str]) -> np.ndarray:
    word_matrix = np.zeros((len(data), len(good_types)), dtype=int)
    word_indices = {}
    n=0
    for word in good_types:
      word_indices[word] = n
      n+=1
    movie_indices = {}
    n=0
    for movie in data:
      movie_indices[movie] = n
      n+=1
    for movie in data:
        for review in data[movie]:
          for token in review["toks"]:
            if token in word_indices.keys():
              word_matrix[movie_indices[movie],word_indices[token]] += 1
    return word_matrix

#Args: index of query movie, movie-word frequency matrix
def similar_movies(movie:int, word_occurence_matrix:np.ndarray):
  scores = []
  for other_movie in range(word_occurence_matrix.shape[0]):
    if not other_movie == movie:
      scores.append(other_movie,\
        np.sum(np.minimum(word_occurence_matrix[movie], word_occurence_matrix[other_movie]))/\
        np.sum(np.maximum(word_occurence_matrix[movie], word_occurence_matrix[other_movie])))
  return sorted(scores, key=lambda x: x[1], reverse=True)[:10]