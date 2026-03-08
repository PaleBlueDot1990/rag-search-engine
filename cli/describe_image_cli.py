import argparse
import mimetypes
import os 
import prompts
from google import genai
from dotenv import load_dotenv
from hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="A text query to rewrite based on the image",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image_bytes = f.read()
    
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")  
    client = genai.Client(api_key=api_key)

    parts = [
        prompts.IMAGE_REWRITE_PROMPT,
        genai.types.Part.from_bytes(data=image_bytes, mime_type=mime),
        args.query,
    ]
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=parts,
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens: {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()