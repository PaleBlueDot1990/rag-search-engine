#!/usr/bin/env python3

import os
import json 
import argparse 
from constants import DOC_LIMIT
from semantic_search import SemanticSearch

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify if the embedding model is loaded")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding of the text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding of the user query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify if documents are embedded succesfully")
    search_query_parser = subparsers.add_parser("search", help="Search most relevant documents for given query")
    search_query_parser.add_argument("query", type=str, help="Query to search documents for")
    search_query_parser.add_argument("--limit", type=int, default=DOC_LIMIT, help="Maximum documents to fetch")

    args = parser.parse_args()
    match args.command:
        case "verify":
            semantic_search = SemanticSearch()
            print(f"Model loaded: {semantic_search.model}")
            print(f"Max sequence length: {semantic_search.model.max_seq_length}")
        case "embed_text":
            semantic_search = SemanticSearch()
            embedding = semantic_search.generate_embedding(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "embedquery":
            semantic_search = SemanticSearch()
            embedding = semantic_search.generate_embedding(args.query)
            print(f"Query: {args.query}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Shape: {embedding.shape}")
        case "verify_embeddings":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                file_data = json.load(f)
            documents = file_data["movies"]
            embeddings = semantic_search.load_or_create_embeddings(documents)
            print(f"Number of docs:   {len(documents)}")
            print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                file_data = json.load(f)
            semantic_search.load_or_create_embeddings(file_data["movies"])

            documents = semantic_search.search(args.query, args.limit)
            for i, doc in enumerate(documents):
                title = doc.get("title", "Unknown Title")
                description = doc.get("description", "No description")
                score = doc.get("score", 0.0)
                print("\n\n\n")
                print(f"{i + 1}. {title}: (score: {score:.4f})")
                print(f"{description}")
                print("\n\n\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()