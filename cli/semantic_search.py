import numpy as np
import os
import constants
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name: str = constants.MODEL) -> None:
        """
        Initialize the SemanticSearch engine with a sentence transformation
        model and empty document storage.

        Attributes:
        model: SentenceTransformer instance for vector encoding.
        embeddings: Numpy array containing generated document vectors.
        documents: List of raw document dictionaries.
        document_map: Mapping from document identifiers to metadata.
        model_name: The identifier of the pre-trained transformer model to be used.
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Generate and persist vector embeddings for a given list of documents by
        concatenating titles and descriptions.

        Parameters:
        documents: A list of movie dictionaries containing 'id', 'title', and 'description'.
        """
        texts = []
        self.documents = documents

        for document in self.documents:
            id = document["id"]
            title = document["title"]
            description = document["description"]

            self.document_map[id] = document
            text = f"{title}: {description}"
            texts.append(text)

        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(os.path.join(constants.CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Load existing embeddings from disk or generate new ones if the cache is
        missing or mismatched with the input document count.

        Parameters:
        documents: A list of movie dictionaries to be indexed or loaded.
        """
        self.documents = documents
        for document in self.documents:
            id = document["id"]
            self.document_map[id] = document

        embeddings_path = os.path.join(constants.CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Produce a vector embedding for a single string of text using the loaded
        transformer model.

        Parameters:
        text: The input string to be encoded.
        """
        if not text or not text.strip():
            raise ValueError("The text is empty or contains only whitespace characters")

        embedding = self.model.encode(text)
        return embedding

    def search(self, query: str, limit: int) -> list[dict]:
        """
        Perform a semantic similarity search across the indexed documents using
        cosine similarity and returns ranked results.

        query: The search string to compare against document embeddings.
        limit: The maximum number of results to return.
        """
        if not os.path.exists(os.path.join(constants.CACHE_DIR, "movie_embeddings.npy")):
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        results = []

        for i in range(0, len(self.documents)):
            doc_embedding = self.embeddings[i]
            similarity_score = float(cosine_similarity(query_embedding, doc_embedding))

            results.append(
                {
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "score": similarity_score,
                }
            )

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
