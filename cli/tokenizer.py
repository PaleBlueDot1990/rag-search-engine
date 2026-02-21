import string
from nltk.stem import PorterStemmer

class Tokenizer:
    def __init__(self) -> None:
        """
        Initialize the tokenizer with a punctuation table, 
        a stemmer, and a stopword list.
        """
        self.table = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()
        with open("data/stopwords.txt", "r") as f:
            file_data = f.read()
        self.stop_words = file_data.splitlines()
    
    def tokenize_sentence(self, text : str) -> list[str]:
        """
        Tokenize a sentence by lower-casing, removing punctuation, 
        filtering stopwords, and stemming.
        """
        text = text.lower()
        text = text.translate(self.table)
        words = text.split()
        tokens = []
        for word in words:
            if word and word.strip() and word not in self.stop_words:
                word = self.stemmer.stem(word)
                tokens.append(word)
        return tokens
    
    def tokenize_word(self, word : str) -> str:
        """
        Tokenize a single word by lowercasing, removing punctuation, 
        filtering stopwords, and stemming.
        """
        word = word.lower()
        word = word.translate(self.table)
        token = ''
        if word and word.strip() and word not in self.stop_words:
            token = self.stemmer.stem(word)
        return token