from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict(list)

    def generate_embeddings(self, text):
        if len(text) == 0 or not text:
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]


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
