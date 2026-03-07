import numpy as np
import prompts
import os
from google import genai
from dotenv import load_dotenv
from inverted_index import InvertedIndex
from tokenizer import Tokenizer
from chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        """
        Initialize the HybridSearch engine by coordinating semantic and
        keyword-based search components.

        Attributes:
        documents: List of raw document dictionaries.
        css: ChunkedSemanticSearch instance for segment-level matching.
        tokenizer: Tokenizer instance for keyword processing.
        index: InvertedIndex instance for BM25 keyword search.
        """
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")  

        self.client = genai.Client(api_key=self.api_key)
        self.documents = documents

        self.css = ChunkedSemanticSearch()
        self.css.load_or_create_chunk_embeddings(documents)

        self.tokenizer = Tokenizer()
        self.index = InvertedIndex(self.tokenizer)
        self.index.load_or_create()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        """
        Internal method to execute a BM25 keyword search.

        Parameters:
        query: The search string.
        limit: Maximum number of results to return.
        """
        return self.index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """
        Execute a hybrid search by combining semantic and BM25 scores using
        a weighted linear combination (alpha * semantic + (1-alpha) * bm25).

        Parameters:
        query: The search string.
        alpha: Weight assigned to semantic search (0.0 to 1.0).
        limit: Maximum number of results to return.
        """
        search_limit = 500 * limit
        bm25_results = self.index.bm25_search(query, search_limit)
        css_results = self.css.search_chunks(query, search_limit)

        bm25_map = {res["id"]: res["score"] for res in bm25_results}
        css_map = {res["id"]: res["score"] for res in css_results}

        relevant_ids = list(set(bm25_map.keys()) | set(css_map.keys()))
        if not relevant_ids:
            return []

        bm25_scores_raw = [bm25_map.get(doc_id, 0.0) for doc_id in relevant_ids]
        css_scores_raw = [css_map.get(doc_id, 0.0) for doc_id in relevant_ids]

        bm25_norm = self.get_normalized_scores(bm25_scores_raw)
        css_norm = self.get_normalized_scores(css_scores_raw)

        hybrid_results = []
        for i, doc_id in enumerate(relevant_ids):
            bm25_norm_score = bm25_norm[i]
            css_norm_score = css_norm[i]
            hybrid_score = self._hybrid_score(bm25_norm_score, css_norm_score, alpha)
            
            doc = next(d for d in self.documents if d["id"] == doc_id)
            hybrid_results.append({
                "id": doc_id,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "bm25_norm_score": bm25_norm_score,
                "css_norm_score": css_norm_score,
                "hybrid_score": hybrid_score,
            })

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:limit]

    def rrf_search(self, query: str, enhance: str, k: int = 60, limit: int = 10) -> list[dict]:
        """
        Execute a hybrid search using Reciprocal Rank Fusion (RRF).

        Parameters:
        query: The search string.
        k: RRF constant (default 60).
        limit: Maximum number of results to return.
        """
        query = self._enhance_query(query, enhance)

        search_limit = 500 * limit
        bm25_results = self.index.bm25_search(query, search_limit)
        css_results = self.css.search_chunks(query, search_limit)

        bm25_ranks = {res["id"]: i + 1 for i, res in enumerate(bm25_results)}
        css_ranks = {res["id"]: i + 1 for i, res in enumerate(css_results)}
        all_doc_ids = set(bm25_ranks.keys()) | set(css_ranks.keys())

        results = []
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id)
            css_rank = css_ranks.get(doc_id)

            rrf_score = 0.0
            if bm25_rank:
                rrf_score += 1.0 / (k + bm25_rank)
            if css_rank:
                rrf_score += 1.0 / (k + css_rank)
            
            doc = next(d for d in self.documents if d["id"] == doc_id)
            title = doc["title"]

            results.append({
                "title": title,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank,
                "css_rank": css_rank,
            })

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:limit]

    def _enhance_query(self, query: str, enhance: str) -> str:
        """
        Internal method to enhance the search query using LLM-based techniques.

        Parameters:
        query: The original search string.
        enhance: The enhancement method ('spell', 'rewrite', 'expand').
        """
        prompt_map = {
            "spell": prompts.SPELL_CHECK_PROMPT,
            "rewrite": prompts.REWRITE_PROMPT,
            "expand": prompts.EXPAND_PROMPT,
        }

        if enhance not in prompt_map:
            return query

        response = self.client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt_map[enhance].format(query=query)
        )

        enhanced_query = response.text
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query

    def get_normalized_scores(self, scores: list[float]) -> list[float]:
        """
        Internal utility to normalize a list of scores using Min-Max scaling.

        Parameters:
        scores: The raw list of floating point scores to be normalized.
        """
        if not scores:
            return []
            
        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0] * len(scores)
        
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        """
        Internal utility to compute the weighted linear combination of two scores.

        Parameters:
        bm25_score: The normalized BM25 relevance score.
        semantic_score: The normalized semantic similarity score.
        alpha: The weight assigned to the bm25 component.
        """
        return (alpha * bm25_score) + ((1 - alpha) * semantic_score)
    
    

