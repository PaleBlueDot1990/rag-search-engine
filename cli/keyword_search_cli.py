#!/usr/bin/env python3

import argparse
import pickle 
from search_helper import SearchHelper
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            test_build_command()
        case "search":
            search_helper = SearchHelper()
            ans = search_helper.get_top_five_movies(args.query)
            print(f"Searching for: {args.query}")
            for i in range(0, len(ans)):
                print(f"{i + 1}. {ans[i]}")
        case _:
            parser.print_help()

def test_build_command() -> None:
    with open("cache/index.pkl", "rb") as f:
        index = pickle.load(f)
        doc_id = list(index["merida"])[0]
        print(f"First document id for token merida = {doc_id}")
        with open("cache/docmap.pkl", "rb") as g:
            docmap = pickle.load(g)
            document = docmap[doc_id]
            print(f"First document for token merida = {document}")

if __name__ == "__main__":
    main() 