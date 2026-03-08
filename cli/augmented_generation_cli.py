import argparse
import json
import constants
import os 
import prompts
from google import genai
from dotenv import load_dotenv
from hybrid_search import HybridSearch

class AugmentedGenerationCLI:
    def __init__(self) -> None:
        """
        Initialize the AugmentedGenerationCLI with lazy-loaded components and
        default data paths.
        """
        self.hybrid_search = None
        self.client = None
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
    
    def _get_genai_client(self) -> genai.Client:
        """
        Internal method to lazy-load and retrieve the Gemini API client.
        """
        if self.client is None:
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")  
            self.client = genai.Client(api_key=api_key)
        return self.client

    def handle_rag(self, args: argparse.Namespace) -> None:
        """
        Command handler to execute a Retrieval-Augmented Generation (RAG) query.

        Parameters:
        args: Parsed arguments containing the search query.
        """
        query = args.query 
        hs = self._get_hybrid_search()
        results = hs.rrf_search(query, None, None, limit=5)

        docs = ""
        for doc in results:
            docs += f"Movie Id: {doc['id']}\n"
            docs += f"Movie Title: {doc['title']}\n"
            docs += f"Movie Description: {doc['description']}\n\n"

        client = self._get_genai_client()
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts.AUGMENT_PROMPT.format(
                query=query, 
                docs=docs, 
            )
        )

        for result in results:
            print(f"- {result['title']}")
        print(f"\nRAG Response:\n{response.text.strip()}")
    
    def handle_summarize(self, args: argparse.Namespace) -> None:
        """
        Command handler to search and summarize the returned documents.

        Parameters:
        args: Parsed arguments containing the search query and limit.
        """
        query = args.query
        limit = args.limit 

        hs = self._get_hybrid_search()
        results = hs.rrf_search(query, None, None, limit=limit)

        docs = ""
        for doc in results:
            docs += f"Movie Id: {doc['id']}\n"
            docs += f"Movie Title: {doc['title']}\n"
            docs += f"Movie Description: {doc['description']}\n\n"

        client = self._get_genai_client()
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts.SUMMARIZE_PROMPT.format(
                query=query, 
                results=docs, 
            )
        )

        for result in results:
            print(f"- {result['title']}")
        print(f"\nLLM Summary:\n{response.text.strip()}")
    
    def handle_citations(self, args: argparse.Namespace) -> None:
        """
        Command handler to search documents and generate a cited answer.

        Parameters:
        args: Parsed arguments containing the search query and limit.
        """
        query = args.query
        limit = args.limit

        hs = self._get_hybrid_search()
        results = hs.rrf_search(query, None, None, limit=limit)

        docs = ""
        for doc in results:
            docs += f"Movie Id: {doc['id']}\n"
            docs += f"Movie Title: {doc['title']}\n"
            docs += f"Movie Description: {doc['description']}\n\n"

        client = self._get_genai_client()
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts.CITATIONS_PROMPT.format(
                query=query, 
                documents=docs, 
            )
        )

        for result in results:
            print(f"- {result['title']}")
        print(f"\nLLM Answer:\n{response.text.strip()}")

    def handle_question(self, args: argparse.Namespace) -> None:
        """
        Command handler to answer a user's question using retrieved documents.

        Parameters:
        args: Parsed arguments containing the question text and limit.
        """
        question = args.question
        limit = args.limit

        hs = self._get_hybrid_search()
        results = hs.rrf_search(question, None, None, limit=limit)

        docs = ""
        for doc in results:
            docs += f"Movie Id: {doc['id']}\n"
            docs += f"Movie Title: {doc['title']}\n"
            docs += f"Movie Description: {doc['description']}\n\n"

        client = self._get_genai_client()
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts.ANSWER_PROMPT.format(
                query=question, 
                context=docs, 
            )
        )

        for result in results:
            print(f"- {result['title']}")
        print(f"\nLLM Answer:\n{response.text.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Augmented generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Search and generate an answer")
    rag_parser.add_argument("query", type=str, help="Search query")

    summarize_parser = subparsers.add_parser("summarize", help="Search and summarize results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Max documents to retrieve")

    citations_parser = subparsers.add_parser("citations", help="Search and answer with citations")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Max documents to retrieve")

    question_parser = subparsers.add_parser("question", help="Answer a question from retrieved docs")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Max documents to retrieve")

    args = parser.parse_args()
    cli = AugmentedGenerationCLI()

    match args.command:
        case "rag":
            cli.handle_rag(args)
        case "summarize":
            cli.handle_summarize(args)
        case "citations":
            cli.handle_citations(args)
        case "question":
            cli.handle_question(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()