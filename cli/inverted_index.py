import os 
import pickle
import json 
from collections import Counter
from tokenizer import Tokenizer

class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer) -> None:
        """
        Initialize the inverted index with an empty index, 
        empty docmap, empty term_frequency and a tokenizer
        """
        self.index = {}            # key : token       → value : set of document IDs that contain the token 
        self.docmap = {}           # key : document ID → value : movie dict (id, title, description)
        self.term_frequencies = {} # key : document ID → value : counter dictionary (provided by python)
        self.tokenizer = tokenizer
    
    def __add_document(self, doc_id : int, text : str) -> None:
        """
        Tokenize the text, update term frequency of given 
        document ID, and add each token to the index with 
        the given document ID.
        """
        tokens = self.tokenizer.tokenize_sentence(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
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
        Persist the index, docmap and term frequencies 
        to disk as pickle files in the cache directory.
        """
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self) -> None:
        """
        Load the cached index, docmap and term frequencies 
        from disk.
        """
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
    
    def get_tf(self, doc_id : int, term : str) -> int:
        """
        Get term frequency of the given term in the given 
        document ID.
        """
        token = self.tokenizer.tokenize_word(term)
        return self.term_frequencies[doc_id][token]


    
