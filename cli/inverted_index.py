import os 
import pickle
import json 
from tokenizer import Tokenizer

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.tokenizer = Tokenizer()
        with open("data/movies.json", "r") as f:
            file_data = json.load(f)
        self.movies = file_data["movies"]
    
    def __add_document(self, doc_id : int, text : str):
        tokens = self.tokenizer.tokenize_sentence(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_document(self, term : str) -> list[int]:
        token = self.tokenizer.tokenize_word(term)
        if token not in self.index:
            return []
        doc_ids = self.index[token]
        return sorted(doc_ids)
    
    def build(self):
        for movie in self.movies:
            id = movie["id"]
            title = movie["title"]
            description = movie["description"]
            self.__add_document(id, f"{title} {description}")
            self.docmap[id] = movie
    
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)


    


    
