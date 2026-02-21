import json
from tokenizer import Tokenizer
from inverted_index import InvertedIndex

class SearchHelper:
    def __init__(self, tokenizer: Tokenizer, inverted_index: InvertedIndex) -> None:
        """
        Initialize the search helper with a tokenizer and movie data.
        """
        self.tokenizer = tokenizer
        self.inverted_index = inverted_index

    def get_top_five_movies(self, query : str) -> list[str]:
        """
        Return the first 5 movies whose titles match the query tokens.
        """
        self.inverted_index.load()
        query_tokens = self.tokenizer.tokenize_sentence(query)

        ans = []
        for qtoken in query_tokens:
            doc_ids = self.inverted_index.get_document(qtoken)
            docs_to_take = min(5 - len(ans), len(doc_ids))
            for i in range(0, docs_to_take):
                doc_id = doc_ids[i]
                movie_title = self.inverted_index.docmap[doc_id]["title"]
                ans.append(movie_title)
            if(len(ans) == 5):
                break
        return ans