import string
from nltk.stem import PorterStemmer


class Tokenizer:
    def __init__(self) -> None:
        """
        Initialize the Tokenizer with a punctuation translation table, a Porter
        stemmer instance, and a loaded list of stopwords.

        No arguments.

        Attributes:
        table: Translation table for stripping punctuation from strings.
        stemmer: PorterStemmer instance for reducing words to their root
        forms.
        stop_words: List of common words to be excluded from tokenization.
        """
        self.table = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()
        with open("data/stopwords.txt", "r") as f:
            file_data = f.read()
        self.stop_words = file_data.splitlines()

    def tokenize_sentence(self, text: str) -> list[str]:
        """
        Process a string into a list of normalized tokens by performing
        lowercase conversion, punctuation removal, stopword filtering, and
        linguistic stemming.

        text: The input string to be tokenized.
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

    def tokenize_word(self, word: str) -> str:
        """
        Process a single word into a normalized token by performing lowercase conversion,
        punctuation removal, stopword filtering, and linguistic stemming.

        word: The individual word to be tokenized.
        """
        word = word.lower()
        word = word.translate(self.table)
        token = ""
        if word and word.strip() and word not in self.stop_words:
            token = self.stemmer.stem(word)
        return token
