import os

BM25_K1 = 1.5
BM25_B = 0.75
DOC_LIMIT = 5
MODEL = "all-MiniLM-L6-v2"
CHUNK_LIMIT = 200
CHUNK_OVERLAP = 40
SEMANTIC_CHUNK_LIMIT = 4
SEMANTIC_CHUNK_OVERLAP = 1

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
