#!/usr/bin/env python3

import argparse
from constants import BM25_K1, BM25_B, DOC_LIMIT
from tokenizer import Tokenizer
from inverted_index import InvertedIndex


class KeywordSearchCLI:
    def __init__(self):
        """
        Initialize the KeywordSearchCLI with lazy-loaded components for
        tokenization and indexing.

        No arguments.

        Attributes:
        tokenizer: Lazily initialized Tokenizer instance.
        inverted_index: Lazily initialized InvertedIndex instance.
        """
        self.tokenizer = None
        self.inverted_index = None

    def _get_resources(self):
        """
        Internal method to lazy-load and retrieve the core search components,
        ensuring they are initialized only when necessary.

        No arguments.
        """
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
        if self.inverted_index is None:
            self.inverted_index = InvertedIndex(self.tokenizer)
        return self.tokenizer, self.inverted_index

    def _get_top_matches(self, query: str) -> list[str]:
        """
        Internal method to retrieve the top 5 document titles matching the
        provided query tokens from the inverted index.

        query: The raw search string to be matched.
        """
        tokenizer, index = self._get_resources()
        index.load()
        query_tokens = tokenizer.tokenize_sentence(query)

        ans = []
        for qtoken in query_tokens:
            doc_ids = index.get_document(qtoken)
            docs_to_take = min(5 - len(ans), len(doc_ids))
            for i in range(0, docs_to_take):
                doc_id = doc_ids[i]
                movie_title = index.docmap[doc_id]["title"]
                ans.append(movie_title)
            if len(ans) == 5:
                break
        return ans

    def handle_build(self, args):
        """
        Command handler to trigger the construction and persistence of the
        inverted index.

        args: The parsed command-line arguments.
        """
        _, index = self._get_resources()
        index.build()
        index.save()
        print("Inverted index successfully built and persisted.")

    def handle_search(self, args):
        """
        Command handler to execute a token-based search and display the top 5
        results.

        args: The parsed command-line arguments containing the search query.
        """
        results = self._get_top_matches(args.query)
        print(f"Results for query: '{args.query}'")
        for i, title in enumerate(results):
            decoded_title = title.encode("utf-8").decode("unicode_escape")
            print(f"{i + 1}. {decoded_title}")

    def handle_tf(self, args):
        """
        Command handler to calculate and display the term frequency (TF) for a
        given word in a specific document.

        args: The parsed command-line arguments containing document ID and term.
        """
        _, index = self._get_resources()
        index.load()
        tf = index.get_tf(args.document_id, args.term)
        print(f"Term Frequency for '{args.term}' in document {args.document_id}: {tf}")

    def handle_idf(self, args):
        """
        Command handler to calculate and display the inverse document frequency
        (IDF) for a term across the corpus.

        args: The parsed command-line arguments containing the term.
        """
        _, index = self._get_resources()
        index.load()
        idf = index.get_idf(args.term)
        print(f"Inverse Document Frequency for '{args.term}': {idf:.4f}")

    def handle_tfidf(self, args):
        """
        Command handler to calculate and display the TF-IDF score for a specific
        term-document pair.

        args: The parsed command-line arguments containing document ID and term.
        """
        _, index = self._get_resources()
        index.load()
        tf = index.get_tf(args.document_id, args.term)
        idf = index.get_idf(args.term)
        score = tf * idf
        print(
            f"TF-IDF score for '{args.term}' in document {args.document_id}: {score:.4f}"
        )

    def handle_bm25tf(self, args):
        """
        Command handler to calculate and display the BM25 term frequency
        component.

        args: The parsed command-line arguments containing document ID, term,
        and BM25 parameters.
        """
        _, index = self._get_resources()
        index.load()
        score = index.get_bm25_tf(args.document_id, args.term, args.k1, args.b)
        print(
            f"BM25 TF score for '{args.term}' in document {args.document_id}: {score:.4f}"
        )

    def handle_bm25idf(self, args):
        """
        Command handler to calculate and display the BM25 inverse document
        frequency component.

        args: The parsed command-line arguments containing the term.
        """
        _, index = self._get_resources()
        index.load()
        score = index.get_bm25_idf(args.term)
        print(f"BM25 IDF score for '{args.term}': {score:.4f}")

    def handle_bm25search(self, args):
        """
        Command handler to execute a BM25-ranked search and display the top
        results.

        args: The parsed command-line arguments containing the query and result
        limit.
        """
        _, index = self._get_resources()
        index.load()
        results = index.bm25_search(args.query, args.limit)
        for i, res in enumerate(results):
            print(
                f"{i + 1}. {res['title']} (score: {res['score']:.4f})\n\n"
            )


def main() -> None:
    """
    Entry point for the Keyword Search CLI, responsible for defining the
    command-line interface, parsing arguments, and dispatching to appropriate
    handlers.

    No arguments.
    """
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define parsers
    subparsers.add_parser("build", help="Generate and persist the inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Execute token-based keyword search"
    )
    search_parser.add_argument("query", type=str, help="Search query string")

    tf_parser = subparsers.add_parser(
        "tf", help="Compute term frequency (TF) for a document"
    )
    tf_parser.add_argument("document_id", type=int, help="Target document identifier")
    tf_parser.add_argument("term", type=str, help="Target term")

    idf_parser = subparsers.add_parser(
        "idf", help="Compute inverse document frequency (IDF)"
    )
    idf_parser.add_argument("term", type=str, help="Target term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Compute TF-IDF score for a document-term pair"
    )
    tfidf_parser.add_argument(
        "document_id", type=int, help="Target document identifier"
    )
    tfidf_parser.add_argument("term", type=str, help="Target term")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Compute BM25 term frequency component"
    )
    bm25_tf_parser.add_argument(
        "document_id", type=int, help="Target document identifier"
    )
    bm25_tf_parser.add_argument("term", type=str, help="Target term")
    bm25_tf_parser.add_argument(
        "--k1", type=float, default=BM25_K1, help="BM25 k1 hyperparameter"
    )
    bm25_tf_parser.add_argument(
        "--b", type=float, default=BM25_B, help="BM25 b hyperparameter"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Compute BM25 inverse document frequency component"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Target term")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Execute BM25-ranked keyword search"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query string")
    bm25search_parser.add_argument(
        "--limit", type=int, default=DOC_LIMIT, help="Maximum results to return"
    )

    args = parser.parse_args()
    cli = KeywordSearchCLI()

    handlers = {
        "build": cli.handle_build,
        "search": cli.handle_search,
        "tf": cli.handle_tf,
        "idf": cli.handle_idf,
        "tfidf": cli.handle_tfidf,
        "bm25tf": cli.handle_bm25tf,
        "bm25idf": cli.handle_bm25idf,
        "bm25search": cli.handle_bm25search,
    }

    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
