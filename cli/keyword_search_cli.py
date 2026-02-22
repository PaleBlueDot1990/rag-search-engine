#!/usr/bin/env python3

import argparse
import pickle 
from constants import BM25_K1, BM25_B, DOC_LIMIT
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
    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a word")
    idf_parser.add_argument("term", type=str, help="The word to check IDF for")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score for a word in a specific document")
    tfidf_parser.add_argument("document_id", type=int, help="The ID of the document (movie id)")
    tfidf_parser.add_argument("term", type=str, help="The word to check TF-IDF for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("document_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=DOC_LIMIT, help="Maximum documents to fetch")

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
            tf = inverted_index.get_tf(args.document_id, args.term)
            print(f"TF for term '{args.term}' in document {args.document_id} is: {tf}")
        case "idf":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            idf = inverted_index.get_idf(args.term)
            print(f"IDF for term '{args.term}' across all documents is: {idf:.2f}")
        case "tfidf":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            tf = inverted_index.get_tf(args.document_id, args.term)
            idf = inverted_index.get_idf(args.term)
            tf_idf = tf * idf 
            print(f"TF-IDF score for term '{args.term}' in document {args.document_id} is: {tf_idf:.2f}")
        case "bm25tf":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            bm25tf = inverted_index.get_bm25_tf(args.document_id, args.term, args.k1, args.b)
            print(f"BM25 TF score for term '{args.term}' in document {args.document_id} is: {bm25tf:.2f}")
        case "bm25idf":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            bm25idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score for term '{args.term}' across all documents is: {bm25idf:.2f}")
        case "bm25search":
            tokenizer = Tokenizer()
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()
            results = inverted_index.bm25_search(args.query, args.limit)
            for i in range(0, len(results)):
                id = results[i]["doc_id"]
                title = results[i]["title"]
                score = results[i]["score"]
                print(f"{i + 1}. ({id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main() 