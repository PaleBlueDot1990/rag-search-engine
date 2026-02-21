#!/usr/bin/env python3

import argparse
import pickle 
from tokenizer import Tokenizer
from search_helper import SearchHelper
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a word in a specific document")
    tf_parser.add_argument("document_id", type=int, help="The ID of the document, that is, movie id")
    tf_parser.add_argument("term", type=str, help="The word for which we are checking frequency for")

    args = parser.parse_args()

    match args.command:
        case "build":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.build()
            inverted_index.save()
        case "search":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            search_helper = SearchHelper(tokenizer, inverted_index)
            ans = search_helper.get_top_five_movies(args.query)
            print(f"Searching for: {args.query}")
            for i in range(0, len(ans)):
                decoded_title = ans[i].encode('utf-8').decode('unicode_escape')
                print(f"{i + 1}. {decoded_title}")
        case "tf":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            ans = inverted_index.get_tf(args.document_id, args.term)
            print(f"TF for term '{args.term}' in document {args.document_id} is: {ans}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 