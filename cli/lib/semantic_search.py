import os
from collections import defaultdict

import numpy as np
from lib.search_utils import CACHE_DIR, load_movies
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embeddings(self, text):
        if len(text) == 0 or not text:
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
            else:
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)
        return self.embeddings

    def build_embeddings(self, documents):
        movies = []
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings


def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
