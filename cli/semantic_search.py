import numpy as np
import os 
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents : list[dict]):
        texts = []
        self.documents = documents

        for document in self.documents:
            id = document["id"]
            title = document["title"]
            description = document["description"]

            self.document_map[id] = document
            text = f"{title}: {description}"
            texts.append(text)
        
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True
        )

        np.save(
            "cache/movie_embeddings.npy", 
            self.embeddings
        )

        return self.embeddings
    
    def load_or_create_embeddings(self, documents : list[dict]):
        self.documents = documents
        for document in self.documents:
            id = document["id"]
            self.document_map[id] = document
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if(len(self.embeddings) == len(documents)):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def generate_embedding(self, text : str):
        if not text or not text.strip():
            raise ValueError("The text is empty or contains only whitespace characters")
        
        embedding = self.model.encode(text)
        return embedding
    
    def search(self, query: str, limit: int) -> list[dict]:
        if not os.path.exists("cache/movie_embeddings.npy"):
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        results = []

        for i in range(0, len(self.documents)):
            doc_embedding = self.embeddings[i]
            similarity_score = float(cosine_similarity(query_embedding, doc_embedding))
            
            results.append({
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "score": similarity_score
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)