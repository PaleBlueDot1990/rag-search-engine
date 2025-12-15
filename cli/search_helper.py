import json 
import string
from nltk.stem import PorterStemmer


table = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def get_top_five_movies(query : str) -> list[str]:
    with open("data/movies.json", "r") as f:
        file_data = json.load(f)
    movies = file_data["movies"]

    with open("data/stopwords.txt", "r") as f:
        file_data = f.read()
    stop_words = file_data.splitlines()

    ans = []
    query_tokens = process_text(query, stop_words)

    for i in range(0, len(movies)): 
        movie_tokens = process_text(movies[i]["title"], stop_words)
        
        for qtoken in query_tokens:
            break_loop = False 
            for mtoken in movie_tokens:
                if qtoken in mtoken:
                    ans.append(movies[i]["title"])
                    break_loop = True 
                    break
            if break_loop:
                break 

        if(len(ans) == 5):
            break  

    return ans


def process_text(text : str, stop_words : list[str]) -> list[str]:
    # Transform uppercase to lowercase 
    text = text.lower()

    # Remove punctuation marks 
    text = text.translate(table)

    # Split text using whitespace into tokens 
    tokens = text.split()

    # Filter tokens that are non-empty and are not stop words, and also stem them 
    tokens = [
        stemmer.stem(t) 
        for t in tokens 
        if (
            t and 
            t.strip() and 
            t not in stop_words
        )
    ]

    return tokens 


    