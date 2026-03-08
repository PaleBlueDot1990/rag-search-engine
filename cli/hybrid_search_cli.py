import argparse
import json
import constants
import os 
import prompts
from google import genai
from dotenv import load_dotenv
from hybrid_search import HybridSearch


class HybridSearchCLI:
    def __init__(self) -> None:
        """
        Initialize the HybridSearchCLI with lazy-loaded search components and
        default data paths.

        Attributes:
        hybrid_search: Lazily initialized HybridSearch instance.
        data_path: File system path to the source movie data.
        """
        self.hybrid_search = None
        self.data_path = "data/movies.json"

    def _load_movies(self) -> list[dict]:
        """
        Internal method to parse and retrieve the movie list from the JSON data
        file.
        """
        with open(self.data_path, "r") as f:
            file_data = json.load(f)
        return file_data["movies"]

    def _get_hybrid_search(self) -> HybridSearch:
        """
        Internal method to lazy-load and retrieve the HybridSearch engine
        instance.
        """
        if self.hybrid_search is None:
            movies = self._load_movies()
            self.hybrid_search = HybridSearch(movies)
        return self.hybrid_search

    def _evaluate_results(self, query: str, results: list[dict]) -> None:
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")  
        self.client = genai.Client(api_key=self.api_key)

        formatted_results = ""
        for doc in results:
            formatted_results += f"Movie Id: {doc['id']}\n"
            formatted_results += f"Movie Title: {doc['title']}\n"
            formatted_results += f"Movie Description: {doc['description']}\n\n"
        
        response = self.client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts.EVALUATE_PROMPT.format(
                query=query, 
                formatted_results=formatted_results, 
            )
        )

        try:
            scores = json.loads(response.text.strip())
            print("\n--- LLM Evaluation ---")
            for i, result in enumerate(results):
                try:
                    title = result["title"]
                    score = scores[i]
                    print(f"{i + 1}. {title}: {score}/3")
                except IndexError:
                    pass
        except json.JSONDecodeError:
            print("Failed to parse LLM JSON response.")


    def handle_normalize(self, args: argparse.Namespace) -> None:
        """
        Command handler to normalize a list of scores.

        Parameters:
        args: Parsed arguments containing the list of scores.
        """
        scores = args.scores
        if not scores:
            print("No scores provided.")
            return
        
        hs = self._get_hybrid_search()
        normalized_scores = hs.get_normalized_scores(scores)

        for score in normalized_scores:
            print(f"* {score:.4f}")

    def handle_weighted_search(self, args: argparse.Namespace) -> None:
        """
        Command handler to execute a weighted hybrid search.

        Parameters:
        args: Parsed arguments containing query, alpha, and limit.
        """
        hs = self._get_hybrid_search()
        results = hs.weighted_search(args.query, args.alpha, args.limit)
    
        for i, doc in enumerate(results):
            title = (
                doc.get("title", "Unknown Title")
                .encode("utf-8")
                .decode("unicode_escape")
            )
            hybrid_score = doc.get("hybrid_score", 0.0)
            bm25_norm_score = doc.get("bm25_norm_score", 0.0)
            css_norm_score = doc.get("css_norm_score", 0.0)

            print(f"{i + 1}. {title}")
            print(f"Hybrid Score: {hybrid_score:.4f}")
            print(f"BM25: {bm25_norm_score:.4f}, Semantic: {css_norm_score:.4f}\n\n")
    
    def handle_rrf_search(self, args: argparse.Namespace) -> None:
        """
        Command handler to execute a rrf hybrid search.

        Parameters:
        args: Parsed arguments containing query, k, and limit, query enhance method, and llm reranking method.
        """

        hs = self._get_hybrid_search()
        results = hs.rrf_search(args.query, args.enhance, args.rerank_method, args.k, args.limit)

        for i, doc in enumerate(results):
            title = (
                doc.get("title", "Unknown Title")
                .encode("utf-8")
                .decode("unicode_escape")
            )
            llm_rank = (i + 1)
            cross_encoder_score = doc.get("cross_encoder_score", 0.0)
            rrf_score = doc.get("rrf_score", 0.0)
            bm25_rank = doc.get("bm25_rank", 0)
            css_rank = doc.get("css_rank", 0)

            print(f"{i + 1}. {title}")
            if args.rerank_method == "individual" or args.rerank_method == "batch":
                print(f"LLM Ranking : {llm_rank}")
            elif args.rerank_method == "cross_encoder":
                print(f"Cross Encoder Score: {cross_encoder_score}")
            print(f"RRF Score: {rrf_score:.4f}")
            print(f"BM25 Rank: {bm25_rank}, Semantic Rank: {css_rank}\n\n")

        if args.evaluate == True:
            self._evaluate_results(args.query, results)


def main() -> None:
    """
    Entry point for the Hybrid Search CLI, responsible for defining the
    command-line interface, parsing arguments, and dispatching to appropriate
    handlers.

    No arguments.
    """
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize command parser
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of floating point scores to normalize",
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Execute a weighted hybrid (semantic + keyword) search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="The search query string"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for bm25 score component (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results to return (default: 5)",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Execute a rrf hybrid (semantic + keyword) search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="The search query string"
    )
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant"
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results to return (default: 5)"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method"
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method"
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the returned results using LLM"
    )

    args = parser.parse_args()
    cli = HybridSearchCLI()

    match args.command:
        case "normalize":
            cli.handle_normalize(args)
        case "weighted-search":
            cli.handle_weighted_search(args)
        case "rrf-search":
            cli.handle_rrf_search(args)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()