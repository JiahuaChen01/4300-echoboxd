import numpy as np
from collections import Counter

def compute_tf(tokens):
    tf = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}

def compute_idf(docs):
    N = len(docs)
    idf = {}
    for doc in docs:
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1
    return {word: np.log(N / df) for word, df in idf.items()}

def compute_tf_idf(tf, idf):
    return {word: tf[word] * idf.get(word, 0) for word in tf}

def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    num = sum(vec1[w] * vec2[w] for w in common)
    denom1 = np.sqrt(sum(v ** 2 for v in vec1.values()))
    denom2 = np.sqrt(sum(v ** 2 for v in vec2.values()))
    return num / (denom1 * denom2) if denom1 and denom2 else 0