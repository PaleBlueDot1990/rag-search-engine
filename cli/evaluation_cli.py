import argparse
import json 
from hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)
    test_documents = data["test_cases"]

    with open("data/movies.json", "r") as f:
            data = json.load(f)
    documents = data["movies"]
    hs = HybridSearch(documents)

    print(f"k={limit}")
    for test_case in test_documents:
        query = test_case["query"]
        results = hs.rrf_search(query, None, None, 60, limit)

        result_titles = set(r["title"] for r in results)
        relevant_titles = set(test_case["relevant_docs"])
        matching_titles = (result_titles & relevant_titles)

        # precision: how much of what you found is relevant?
        precision_ratio = (
            len(matching_titles) / len(result_titles) 
            if len(result_titles) > 0 else 0.0
        )

        # recall: how much of what's relevant did you find?
        recall_ratio = (
            len(matching_titles) / len(relevant_titles) 
            if len(relevant_titles) > 0 else 0.0
        )

        f1_score = (
            (2 * precision_ratio * recall_ratio) / (precision_ratio + recall_ratio) 
            if (precision_ratio + recall_ratio) > 0 else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_ratio:.4f}")
        print(f"  - Recall@{limit}: {recall_ratio:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {', '.join(result_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}\n")


if __name__ == "__main__":
    main()