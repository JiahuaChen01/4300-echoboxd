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

def default_ins_cost_func(query: str, i: int) -> int:
  return 1
  """
  #starting a new word is more expensive
  if query[i-1] == " ":
    return 2
  else:
    return 1
  """
  
def default_del_cost_func(message: str, i: int) -> int:
  return 1

def default_sub_cost_func(s1: str, s2: str, i: int, j: int) -> float:
  if(s1[i-1] == s2[j-1]):
    return 0
  elif(s1[i-1],s2[j-1]) or (s2[j-1],s1[i-1]) in adj_chars:
    return 1
  else:
    return 2
  
#Args: string of starting query, string of target movie, insertion-cost function, deletion-cost function, substitution-cost function.
def edit_distance(query: str, movie: str, ins_cost_func=default_ins_cost_func, del_cost_func=default_del_cost_func, sub_cost_func=default_sub_cost_func) -> float:
    query = query.lower()
    movie = movie.lower()
    table = np.zeros((len(query)+1,len(movie)+1))
    i = 1 
    for _ in query:
      table[i,0] = table[i-1,0] + del_cost_func(query, i)
      i+=1
    i = 1
    for _ in movie:
      table[0,i] = table[0,i-1] + ins_cost_func(movie, i)
      i+=1
    for i in range(1,len(query)+1):
        for j in range(1,len(movie)+1):
            temp = (
                table[i-1,j] + del_cost_func(query, i),
                table[i,j-1] + ins_cost_func(movie, j),
                table[i-1,j-1] + sub_cost_func(query, movie, i, j),
            )
            table[i,j] = min(temp)
        if np.min(table[i]) > 5:
            return -1
    return table[len(query),len(movie)]


adj_chars = [
    ("a", "q"),
    ("a", "s"),
    ("a", "z"),
    ("b", "g"),
    ("b", "m"),
    ("b", "n"),
    ("b", "v"),
    ("c", "d"),
    ("c", "v"),
    ("c", "x"),
    ("d", "c"),
    ("d", "e"),
    ("d", "f"),
    ("d", "s"),
    ("e", "d"),
    ("e", "r"),
    ("e", "w"),
    ("f", "d"),
    ("f", "g"),
    ("f", "r"),
    ("f", "v"),
    ("g", "b"),
    ("g", "f"),
    ("g", "h"),
    ("g", "t"),
    ("h", "g"),
    ("h", "j"),
    ("h", "m"),
    ("h", "n"),
    ("h", "y"),
    ("i", "k"),
    ("i", "o"),
    ("i", "u"),
    ("j", "h"),
    ("j", "k"),
    ("j", "u"),
    ("k", "i"),
    ("k", "j"),
    ("k", "l"),
    ("l", "k"),
    ("l", "o"),
    ("m", "b"),
    ("m", "h"),
    ("n", "b"),
    ("n", "h"),
    ("o", "i"),
    ("o", "l"),
    ("o", "p"),
    ("p", "o"),
    ("q", "a"),
    ("q", "w"),
    ("r", "e"),
    ("r", "f"),
    ("r", "t"),
    ("s", "a"),
    ("s", "d"),
    ("s", "w"),
    ("s", "x"),
    ("t", "g"),
    ("t", "r"),
    ("t", "y"),
    ("u", "i"),
    ("u", "j"),
    ("u", "y"),
    ("v", "b"),
    ("v", "c"),
    ("v", "f"),
    ("w", "e"),
    ("w", "q"),
    ("w", "s"),
    ("x", "c"),
    ("x", "s"),
    ("x", "z"),
    ("y", "h"),
    ("y", "t"),
    ("y", "u"),
    ("z", "a"),
    ("z", "x"),
]
