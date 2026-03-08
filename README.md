# RAG Search Engine

A movie search engine that combines classical keyword retrieval, dense semantic search, and LLM-powered generation — all over a 'hypothetical' Netflix movie dataset.

## Features

- **BM25 Keyword Search** — Inverted index with Porter stemming, stopword filtering, and BM25 scoring
- **Semantic Search** — Dense vector search using `all-MiniLM-L6-v2` with document-level and chunk-level modes
- **Hybrid Search** — Combines BM25 + semantic results via Weighted scoring or Reciprocal Rank Fusion (RRF)
- **LLM Reranking** — Three strategies: individual scoring, batch ranking, and CrossEncoder neural reranking
- **Query Enhancement** — LLM-powered spell check, query rewriting, and query expansion before search
- **RAG Generation** — Four LLM generation modes over retrieved documents: RAG, Summarize, Citations, Q&A
- **Multimodal Search** — CLIP-based image embedding to find movies visually similar to a given image
- **Image Query Rewriting** — Uses Gemini Vision to rewrite a text query based on an image
- **Evaluation** — Precision@k, Recall@k, and F1 scoring against a golden dataset

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── inverted_index.py          # BM25 inverted index implementation
│   ├── tokenizer.py               # Tokenizer (stemming, stopwords)
│   ├── semantic_search.py         # Dense embedding search base class
│   ├── chunked_semantic_search.py # Chunk-level semantic search
│   ├── hybrid_search.py           # Hybrid search engine (weighted + RRF)
│   ├── prompts.py                 # All LLM prompt templates
│   ├── constants.py               # Hyperparameters and config
│   ├── keyword_search_cli.py      # CLI: BM25 keyword search
│   ├── semantic_search_cli.py     # CLI: Semantic and chunk search
│   ├── hybrid_search_cli.py       # CLI: Hybrid search with reranking
│   ├── augmented_generation_cli.py# CLI: LLM generation (RAG, summarize, Q&A)
│   ├── evaluation_cli.py          # CLI: Precision/Recall/F1 evaluation
│   ├── describe_image_cli.py      # CLI: Multimodal query rewriting
│   └── multimodal_search_cli.py   # CLI: CLIP-based image search
├── data/
│   ├── movies.json                # Netflix movie corpus (~26MB)
│   ├── golden_dataset.json        # Labeled queries for evaluation
│   ├── stopwords.txt              # Stopword list
│   └── paddington.jpeg            # Sample image for testing
└── cache/                         # Auto-generated index and embedding cache
```

## Setup

**Requirements**: Python 3.13+, [`uv`](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Usage

All commands are run from the project root with `uv run cli/<script>.py`.

### Keyword Search (BM25)
```bash
uv run cli/keyword_search_cli.py search "bear survival wilderness"
```

### Semantic Search
```bash
uv run cli/semantic_search_cli.py search "animated bear family movie"
uv run cli/semantic_search_cli.py chunk-search "bear in the woods"
```

### Hybrid Search (RRF)
```bash
# Plain RRF
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --limit 5

# With query enhancement
uv run cli/hybrid_search_cli.py rrf-search "bera movie kids" --enhance spell

# With LLM reranking
uv run cli/hybrid_search_cli.py rrf-search "bear wilderness survival" --rerank-method batch --limit 5

# With LLM evaluation of results
uv run cli/hybrid_search_cli.py rrf-search "bear in the woods" --limit 5 --evaluate
```

### RAG Generation
```bash
uv run cli/augmented_generation_cli.py rag "what bear movies are on netflix?"
uv run cli/augmented_generation_cli.py summarize "bear survival movies" --limit 5
uv run cli/augmented_generation_cli.py citations "grizzly bear documentaries"
uv run cli/augmented_generation_cli.py question "What is the Revenant about?"
```

### Evaluation
```bash
uv run cli/evaluation_cli.py --limit 6
```

### Image-Based Search
```bash
# Rewrite a text query using an image
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear movie"

# Search for movies visually similar to an image
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

## Configuration

Edit `cli/constants.py` to tune search behaviour:

| Constant | Default | Description |
|---|---|---|
| `MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for semantic search |
| `BM25_K1` | `1.5` | BM25 term saturation parameter |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `SEMANTIC_CHUNK_LIMIT` | `4` | Sentences per chunk |
| `SEMANTIC_CHUNK_OVERLAP` | `1` | Overlapping sentences between chunks |

## Dependencies

- [`sentence-transformers`](https://www.sbert.net/) — Semantic embeddings and CrossEncoder reranking
- [`google-genai`](https://ai.google.dev/) — Gemini API for LLM generation and reranking
- [`nltk`](https://www.nltk.org/) — Porter stemmer for BM25 tokenization
- [`Pillow`](https://pillow.readthedocs.io/) — Image loading for multimodal search
- [`numpy`](https://numpy.org/) — Vector operations and embedding storage
