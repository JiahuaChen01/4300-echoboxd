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

#Args: string of starting query, string of target movie, insertion-cost funciton, deletion-cost function, substitution-cost function.
def edit_distance(query: str, movie: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int) -> int:
    query = query.lower()
    message = message.lower()
    table = np.zeros((len(query)+1,len(message)+1))
    i = 1 
    for _ in query:
      table[i,0] = table[i-1,0] + del_cost_func(query, i)
      i+=1
    i = 1
    for _ in message:
      table[0,i] = table[0,i-1] + ins_cost_func(message, i)
      i+=1
    for i in range(1,len(query)+1):
      for j in range(1,len(message)+1):
        table[i,j] = min(
          table[i-1,j] + del_cost_func(query, i),
          table[i,j-1] + ins_cost_func(message, j),
          table[i-1,j-1] + sub_cost_func(query, message, i, j),
        )
    return table[len(query),len(message)]

def ins_cost_func(string: str, i: int):
  if string[i] == " ":
    return 2
  else:
    return 1
  
def del_cost_func(string: str, i: int):
  return 1

def sub_cost_func(s1: str, s2: str, i: int, j: int):
  return 1