import os
import pickle
import json
import math
import constants
from collections import Counter
from tokenizer import Tokenizer


class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer) -> None:
        """
        Initialize the InvertedIndex with essential data structures including
        the token-to-document index, document metadata map, term frequencies,
        and document lengths.

        tokenizer: An instance of the Tokenizer class used for text processing.

        Attributes:
        index: Dictionary mapping tokens to sets of document identifiers.
        docmap: Dictionary mapping document identifiers to movie metadata.
        term_frequencies: Dictionary mapping document IDs to token counters.
        doc_lengths: Dictionary mapping document identifiers to token counts.
        tokenizer: Reference to the Tokenizer instance for text processing.
        """
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.tokenizer = tokenizer

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Internal method to process and integrate a document into the index by
        tokenizing text, updating term frequencies, and recording lengths.

        doc_id: The unique integer identifier for the document.
        text: The raw text content of the document to be indexed.
        """
        tokens = self.tokenizer.tokenize_sentence(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        """
        Calculate the mean length of all indexed documents, utilizing a cached
        value if available to optimize performance.

        No arguments.
        """
        if hasattr(self, "avg_doc_length"):
            return self.avg_doc_length

        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / len(self.doc_lengths)
        return self.avg_doc_length

    def get_document(self, term: str) -> list[int]:
        """
        Retrieve a sorted list of document identifiers associated with a
        specific search term.

        term: The search term to be queried in the index.
        """
        token = self.tokenizer.tokenize_word(term)
        if token not in self.index:
            return []
        doc_ids = self.index[token]
        return sorted(doc_ids)

    def build(self) -> None:
        """
        Construct the complete inverted index and document metadata map by
        processing the local movie datasets.

        No arguments.
        """
        with open("data/movies.json", "r") as f:
            file_data = json.load(f)
        movies = file_data["movies"]

        for movie in movies:
            id = movie["id"]
            title = movie["title"]
            description = movie["description"]
            self.__add_document(id, f"{title} {description}")
            self.docmap[id] = movie

    def save(self) -> None:
        """
        Serialize and persist the current state of the index, document map, and
        statistical data to serialized binary files in the cache directory.

        No arguments.
        """
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """
        Deserialize and load the index, document map, and statistical data from
        binary storage into memory.

        No arguments.
        """
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Retrieve the raw frequency of a specific term within a specified
        document.

        doc_id: The identifier of the target document.
        term: The term whose frequency is to be retrieved.
        """
        token = self.tokenizer.tokenize_word(term)
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        """
        Calculate the Inverse Document Frequency (IDF) score for a given term
        across the entire document corpus.

        term: The term for which the IDF score is calculated.
        """
        token = self.tokenizer.tokenize_word(term)
        if token in self.index:
            term_match_doc_count = len(self.index[token])
        else:
            term_match_doc_count = 0

        total_doc_count = len(self.docmap)
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        """
        Calculate the BM25-normalized term frequency component for a specific
        term and document using specified hyperparameters.

        doc_id: The identifier of the document.
        term: The target term for calculation.
        k1: The BM25 saturation parameter.
        b: The BM25 length normalization parameter.
        """
        token = self.tokenizer.tokenize_word(term)
        tf = self.term_frequencies[doc_id][token]
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        bm25 = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate the BM25-specific Inverse Document Frequency component for
        a given term.

        term: The term for which the BM25 IDF score is calculated.
        """
        token = self.tokenizer.tokenize_word(term)
        if token in self.index:
            term_match_doc_count = len(self.index[token])
        else:
            term_match_doc_count = 0

        total_doc_count = len(self.docmap)

        numerator = total_doc_count - term_match_doc_count + 0.5
        denominator = term_match_doc_count + 0.5
        return math.log(numerator / denominator + 1)

    def get_bm25(self, doc_id: int, term: str) -> float:
        """
        Compute the integrated BM25 relevance score for a term within a
        document by combining normalized frequency and IDF components.

        doc_id: The identifier of the document.
        term: The target term.
        """
        bm25_tf = self.get_bm25_tf(doc_id, term, constants.BM25_K1, constants.BM25_B)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        """
        Execute a search using the BM25 scoring algorithm, returning a set of
        ranked document results based on query token relevance.

        query: The raw search string.
        limit: The maximum number of results to return.
        """
        scores = {}
        tokens = self.tokenizer.tokenize_sentence(query)

        for token in tokens:
            if token not in self.index:
                continue
            for doc_id in self.index[token]:
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += self.get_bm25(doc_id, token)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            results.append(
                {
                    "doc_id": doc_id,
                    "title": self.docmap[doc_id]["title"],
                    "score": score,
                }
            )
        return results
