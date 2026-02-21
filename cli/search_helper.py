import json
from tokenizer import Tokenizer

class SearchHelper:
    def __init__(self):
        """Initialize the search helper with a tokenizer and movie data."""
        self.tokenizer = Tokenizer()
        with open("data/movies.json", "r") as f:
            file_data = json.load(f)
        self.movies = file_data["movies"]

    def get_top_five_movies(self, query : str) -> list[str]:
        """Return the first 5 movies whose titles match the query tokens."""
        ans = []
        query_tokens = self.tokenizer.tokenize_sentence(query)
        for i in range(0, len(self.movies)):
            movie_tokens = self.tokenizer.tokenize_sentence(self.movies[i]["title"])
            if self.__is_query_token_in_movie_token(query_tokens, movie_tokens):
                ans.append(self.movies[i]["title"])
            if(len(ans) == 5):
                break
        return ans
    
    def __is_query_token_in_movie_token(self, query_tokens : list[str], movie_tokens : list[str]) -> bool:
        """Check if any query token is a substring of any movie title token."""
        for qtoken in query_tokens:
            for mtoken in movie_tokens:
                if qtoken in mtoken:
                    return True
        return False