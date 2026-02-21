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
        Initialize the inverted index with an empty index, 
        empty docmap, empty term_frequency, empty doc_lengths
        and a tokenizer.
        """
        self.index = {}            # key : token       → value : set of document IDs that contain the token 
        self.docmap = {}           # key : document ID → value : movie dict (id, title, description)
        self.term_frequencies = {} # key : document ID → value : counter dictionary (provided by python)
        self.doc_lengths = {}      # key : document ID → value : length of the document 
        self.tokenizer = tokenizer
    
    def __add_document(self, doc_id : int, text : str) -> None:
        """
        Tokenize the text, update term frequency and document 
        length of given document ID, and add each token to 
        the index with the given document ID.
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
        Calculate and cache the average length of all documents 
        currently in the index.
        """
        if hasattr(self, 'avg_doc_length'):
            return self.avg_doc_length
            
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / len(self.doc_lengths)
        return self.avg_doc_length
    
    def get_document(self, term : str) -> list[int]:
        """Return a sorted list of document IDs that 
        contain the given term.
        """
        token = self.tokenizer.tokenize_word(term)
        if token not in self.index:
            return []
        doc_ids = self.index[token]
        return sorted(doc_ids)
    
    def build(self) -> None:
        """Build the inverted index and docmap from the 
        loaded movie data.
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
        Persist the index, docmap, term frequencies and 
        document lengths to disk as pickle files in the 
        cache directory.
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
        Load the cached index, docmap, term frequencies and 
        document lengths from disk.
        """
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    def get_tf(self, doc_id : int, term : str) -> int:
        """
        Get term frequency of the given term in the given 
        document ID.
        """
        token = self.tokenizer.tokenize_word(term)
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term : str) -> float:
        """
        Get inverse document frequency of the given term 
        across all documents.
        """
        token = self.tokenizer.tokenize_word(term)
        if token in self.index:
            term_match_doc_count = len(self.index[token])
        else:
            term_match_doc_count = 0
        
        total_doc_count = len(self.docmap)
        return math.log(
            (total_doc_count + 1) / (term_match_doc_count + 1)
        )
    
    def get_bm25_tf(self, doc_id : int, term : str, k1 : float, b : float) -> float:
        """
        Get bm25 term frequency of the given term in the given 
        document ID.
        """
        token = self.tokenizer.tokenize_word(term)
        tf = self.term_frequencies[doc_id][token]
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        bm25 = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25 
    
    def get_bm25_idf(self, term : str) -> float:
        """
        Get bm25 inverse document frequency of the given term 
        across all documents.
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
    
