import constants
import json
import numpy as np
import re
import os
from semantic_search import SemanticSearch

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = constants.MODEL) -> None:
        """
        Initialize the ChunkedSemanticSearch engine.

        Attributes:
        chunk_embeddings: Numpy array containing generated chunk vectors.
        chunk_metadata: List of metadata dictionaries for each chunk.
        model_name: The identifier of the pre-trained transformer model to be used.
        """
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Partition documents into overlapping segments and generate vector
        embeddings for each segment. Results are cached to the project root.

        Parameters:
        documents: A list of movie dictionaries containing 'id', 'title' and 'description'.
        """
        self.documents = documents
        all_chunks = []
        chunk_metadata = []

        for document in self.documents:
            doc_id = document["id"]
            title = document["title"]
            description = document["description"]
            self.document_map[doc_id] = document

            if not description or len(description.strip()) == 0:
                continue

            sz = constants.SEMANTIC_CHUNK_LIMIT
            ov = constants.SEMANTIC_CHUNK_OVERLAP
            raw_sentences = re.split(r"(?<=[.!?])\s+", description.strip())
            sentences = [s.strip() for s in raw_sentences if s.strip()]
            doc_chunks = []
            
            for i in range(0, len(sentences), sz - ov):
                doc_chunk = " ".join(sentences[i : i + sz])
                doc_chunks.append(doc_chunk)
                if i + sz >= len(sentences):
                    break

            for idx, doc_chunk in enumerate(doc_chunks):
                chunk_metadata.append({
                    "movie_idx": doc_id,
                    "chunk_idx": idx,
                    "total_chunks": len(doc_chunks),
                })
                all_chunks.append(doc_chunk)

        self.chunk_metadata = chunk_metadata
        os.makedirs(constants.CACHE_DIR, exist_ok=True)
        metadata_path = os.path.join(constants.CACHE_DIR, "chunk_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(all_chunks),
                    "total_movies": len(self.documents),
                    "chunk_size": constants.SEMANTIC_CHUNK_LIMIT,
                    "chunk_overlap": constants.SEMANTIC_CHUNK_OVERLAP
                },
                f,
                indent=2
            )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        embeddings_path = os.path.join(constants.CACHE_DIR, "chunk_embeddings.npy")
        np.save(embeddings_path, self.chunk_embeddings)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Attempt to load existing chunk embeddings and metadata from the cache.
        If the cache is missing, stale, or parameters have changed, it triggers
        a full rebuild.

        Parameters:
        documents: A list of movie dictionaries for validation.
        """
        self.documents = documents
        for document in self.documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document
        
        os.makedirs(constants.CACHE_DIR, exist_ok=True)
        metadata_path = os.path.join(constants.CACHE_DIR, "chunk_metadata.json")
        embeddings_path = os.path.join(constants.CACHE_DIR, "chunk_embeddings.npy")

        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return self.build_chunk_embeddings(self.documents)

        with open(metadata_path, 'r') as f:
            data = json.load(f)

            if (data.get("total_movies") != len(self.documents) or
                data.get("chunk_size") != constants.SEMANTIC_CHUNK_LIMIT or
                data.get("chunk_overlap") != constants.SEMANTIC_CHUNK_OVERLAP):
                print("Cache mismatch or stale parameters detected. Regenerating...")
                return self.build_chunk_embeddings(self.documents)
                
            self.chunk_metadata = data["chunks"]

        self.chunk_embeddings = np.load(embeddings_path)
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """
        Execute a semantic search by comparing the query against individual
        document segments. Scores are aggregated by movie using a Max-Score
        strategy.

        Parameters:
        query: The search string.
        limit: Maximum number of results to return.
        """
        if not query or len(query.strip()) == 0:
            return []
        query_embedding = self.generate_embedding(query.strip())

        movie_scores = {}
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = float(cosine_similarity(query_embedding, chunk_embedding))
            movie_idx = self.chunk_metadata[i]["movie_idx"]
            
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        results = []
        for movie_idx, score in movie_scores.items():
            doc = self.document_map.get(movie_idx)
            results.append({
                "id": movie_idx,
                "title": doc["title"],
                "description": doc["description"],
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two numeric vectors.

    Parameters:
    vec1: The first vector for comparison.
    vec2: The second vector for comparison.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
