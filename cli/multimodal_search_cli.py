import argparse
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearchCLI:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        """
        Initialize the MultimodalSearchCLI with a CLIP SentenceTransformer model.

        Parameters:
        model_name: Name of the SentenceTransformer model to load.
        """
        with open("data/movies.json", "r") as f:
            file_data = json.load(f)
        self.docs = file_data["movies"]

        self.texts = []
        for doc in self.docs:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def _embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate a CLIP embedding for the image at the given path.

        Parameters:
        image_path: File system path to the image.
        """
        image = Image.open(image_path)
        embedding = self.model.encode(image)
        return embedding
    
    def handle_verify_image_embedding(self, args: argparse.Namespace) -> None:
        """
        Command handler to verify that an image can be successfully embedded.

        Parameters:
        args: Parsed arguments containing the image path.
        """
        embedding = self._embed_image(args.image_path)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
    
    def handle_image_search(self, args: argparse.Namespace) -> None:
        """
        Command handler to search for similar movies using an image embedding.

        Parameters:
        args: Parsed arguments containing the image path.
        """
        image_path = args.image_path
        image_embedding = self._embed_image(image_path)

        scores = []
        for i, text_embedding in enumerate(self.text_embeddings):
            dot_product = np.dot(image_embedding, text_embedding)
            norm = np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
            similarity = dot_product / norm if norm > 0 else 0.0
            scores.append((i, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = scores[:5]

        for i, score in top_5:
            doc = self.docs[i]
            title = doc["title"]
            description = doc["description"]
            print(f"{i + 1}. {title} (similarity: {score:.3f})")
    

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding dimensions")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser("image_search", help="Search for documents that are relevant to the image")
    image_search_parser.add_argument("image_path", type=str, help="Path to image")

    args = parser.parse_args()
    cli = MultimodalSearchCLI()

    match args.command:
        case "verify_image_embedding":
            cli.handle_verify_image_embedding(args)
        case "image_search":
            cli.handle_image_search(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

