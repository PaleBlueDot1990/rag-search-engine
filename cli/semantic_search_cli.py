#!/usr/bin/env python3

import re
import json
import argparse
import constants
from semantic_search import SemanticSearch


class SemanticSearchCLI:
    def __init__(self):
        """
        Initialize the SemanticSearchCLI with lazy-loaded search components and
        default data paths.

        No arguments.

        Attributes:
        semantic_search: Lazily initialized SemanticSearch instance.
        data_path: File system path to the source movie data.
        """
        self.semantic_search = None
        self.data_path = "data/movies.json"

    def _get_semantic_search(self):
        """
        Internal method to lazy-load and retrieve the SemanticSearch engine
        instance.

        No arguments.
        """
        if self.semantic_search is None:
            self.semantic_search = SemanticSearch()
        return self.semantic_search

    def _load_movies(self):
        """
        Internal method to parse and retrieve the movie list from the JSON data
        file.

        No arguments.
        """
        with open(self.data_path, "r") as f:
            file_data = json.load(f)
        return file_data["movies"]

    def _format_result(self, i, doc):
        """
        Internal method to format search result metadata into a human-readable
        string for console output.

        i: The index of the search result.
        doc: The document dictionary containing title, description, and
        similarity score.
        """
        title = (
            doc.get("title", "Unknown Title").encode("utf-8").decode("unicode_escape")
        )
        description = (
            doc.get("description", "No description")
            .encode("utf-8")
            .decode("unicode_escape")
        )
        score = doc.get("score", 0.0)
        return f"\n{i + 1}. {title} (score: {score:.4f})\n{description}\n"

    def handle_verify(self, args):
        """
        Command handler to display the status and configuration of the loaded
        embedding model.

        args: Parsed command-line arguments.
        """
        ss = self._get_semantic_search()
        print(f"Model loaded: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")

    def handle_embed_text(self, args):
        """
        Command handler to generate and display vector dimensions for a raw
        string of text.

        args: Parsed arguments containing the target text.
        """
        ss = self._get_semantic_search()
        embedding = ss.generate_embedding(args.text)
        print(f"Text: {args.text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")

    def handle_embed_query(self, args):
        """
        Command handler to generate and display vector properties for a
        specific search query.

        args: Parsed arguments containing the query string.
        """
        ss = self._get_semantic_search()
        embedding = ss.generate_embedding(args.query)
        print(f"Query: {args.query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}")

    def handle_verify_embeddings(self, args):
        """
        Command handler to validate the integrity and shape of the stored
        document embeddings.

        args: Parsed command-line arguments.
        """
        ss = self._get_semantic_search()
        movies = self._load_movies()
        embeddings = ss.load_or_create_embeddings(movies)
        print(f"Number of docs:   {len(movies)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )

    def handle_search(self, args):
        """
        Command handler to execute a semantic similarity search across the
        document corpus.

        args: Parsed arguments containing the query and result limit.
        """
        ss = self._get_semantic_search()
        movies = self._load_movies()
        ss.load_or_create_embeddings(movies)

        results = ss.search(args.query, args.limit)
        for i, doc in enumerate(results):
            print(self._format_result(i, doc))

    def handle_chunk(self, args):
        """
        Command handler to partition text into overlapping fixed-size token
        segments.

        args: Parsed arguments containing text, chunk size, and overlap count.
        """
        sz = args.chunk_size
        ov = args.overlap
        if ov >= sz:
            print("Error: --overlap must be less than --chunk-size")
            return

        words = args.text.strip().split()
        chunks = []

        for i in range(0, len(words), sz - ov):
            chunk = " ".join(words[i : i + sz])
            chunks.append(chunk)
            if i + sz >= len(words):
                break

        print(f"Chunking {len(args.text)} characters into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"{i + 1}. {chunk}")

    def handle_semantic_chunk(self, args):
        """
        Command handler to partition text into segments based on sentence
        boundaries and semantic coherence.

        args: Parsed arguments containing text, max chunk size, and sentence
        overlap.
        """
        sz = args.max_chunk_size
        ov = args.overlap
        if ov >= sz:
            print("Error: --overlap must be less than --max-chunk-size")
            return

        sentences = re.split(r"(?<=[.!?])\s+", args.text.strip())
        chunks = []

        for i in range(0, len(sentences), sz - ov):
            chunk = " ".join(sentences[i : i + sz])
            chunks.append(chunk)
            if i + sz >= len(sentences):
                break

        print(
            f"Semantically chunking {len(args.text)} characters into {len(chunks)} chunks:"
        )
        for i, chunk in enumerate(chunks):
            print(f"{i + 1}. {chunk}")


def main():
    """
    Entry point for the Semantic Search CLI, responsible for defining the
    command-line interface, parsing arguments, and dispatching to appropriate
    handlers.

    No arguments.
    """
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define parsers
    subparsers.add_parser("verify", help="Verify embedding model status")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate vector embedding for input text"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Input string for embedding generation"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate vector embedding for search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Search query string")

    subparsers.add_parser(
        "verify_embeddings", help="Verify integrity of document embeddings"
    )

    search_parser = subparsers.add_parser(
        "search", help="Execute semantic search against the document index"
    )
    search_parser.add_argument("query", type=str, help="Semantic search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=constants.DOC_LIMIT,
        help="Maximum number of results to return",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Partition text into fixed-size segments"
    )
    chunk_parser.add_argument("text", type=str, help="Input text for segmentation")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=constants.CHUNK_LIMIT,
        help="Number of tokens per segment",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=constants.CHUNK_OVERLAP,
        help="Number of overlapping tokens between segments",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Partition text into semantically coherent segments"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Input text for semantic segmentation"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=constants.SEMANTIC_CHUNK_LIMIT,
        help="Maximum number of sentences per segment",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=constants.SEMANTIC_CHUNK_OVERLAP,
        help="Number of overlapping sentences between segments",
    )

    args = parser.parse_args()
    cli = SemanticSearchCLI()

    handlers = {
        "verify": cli.handle_verify,
        "embed_text": cli.handle_embed_text,
        "embedquery": cli.handle_embed_query,
        "verify_embeddings": cli.handle_verify_embeddings,
        "search": cli.handle_search,
        "chunk": cli.handle_chunk,
        "semantic_chunk": cli.handle_semantic_chunk,
    }

    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
