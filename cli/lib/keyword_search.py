import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequency: defaultdict[str, Counter] = defaultdict(Counter)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequency[token][doc_id] += 1

    def get_document(self, term) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequency, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_freq_path, "rb") as f:
                self.term_frequency = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Index or docmap file not found")

    def get_tf(self, term: str, doc_id: int) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequency[token][doc_id]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        idf = math.log((len(self.docmap) + 1) / (len(self.index[token]) + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        df = len(self.index[token])
        n = len(self.docmap)
        bm25_idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def __del__(self):
        self.save()


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    results = []
    query_tokens = tokenize_text(query)
    for query_token in query_tokens:
        doc_tokens = idx.get_document(query_token)
        if doc_tokens:
            for doc_token in doc_tokens:
                results.append(idx.docmap[doc_token])
                if len(results) >= limit:
                    break
    return results


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    bm25_idf = idx.get_bm25_idf(term)

    return bm25_idf


def bm25_tf_command(term: str, doc_id: int) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    tf = idx.get_tf(term, doc_id)
    idf = idx.get_idf(term)
    return tf * idf


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    tf = idx.get_tf(term, doc_id)
    idf = idx.get_idf(term)
    return tf * idf


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    return idx.get_idf(term)


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found")
        exit(1)
    return idx.get_tf(term, doc_id)


# def has_matching_token(query_tokens: list[str], idx: InvertedIndex) -> bool:
#     for query_token in query_tokens:
#         doc_tokens = idx.get_document(query_token)
#         if doc_token:
#             return True
#     return False


def preprocess_text(query: str) -> str:
    lower_text = query.strip().lower()
    clean = lower_text.translate(str.maketrans("", "", string.punctuation))
    return clean


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    stop_words = load_stopwords()
    tokens = text.split()
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if token]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha()]
    return tokens
