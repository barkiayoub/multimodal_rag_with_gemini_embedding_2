"""
ingest.py — One-time script to process the PDF manual and store embeddings in Pinecone.

Strategy: For each page, we combine the extracted text + a rendered image of the page
into a single multimodal embedding using gemini-embedding-2-preview.
This produces one rich 1536-dim vector per page that captures both visual and textual content.

Usage: python ingest.py
"""

import os
import time
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "manual-rag")
PDF_PATH = os.getenv("PDF_PATH", "manual/Documents/docc_vocum.pdf")
PAGES_CACHE_DIR = Path("pages_cache")
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 1536


def setup_pinecone_index(pc: Pinecone) -> object:
    """Create the Pinecone index if it doesn't exist, then return it."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print("  Index ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)


def embed_page(client: genai.Client, page_text: str, image_bytes: bytes) -> list[float]:
    """
    Create a multimodal embedding for a single PDF page.
    Combines text + image into one aggregated vector using gemini-embedding-2-preview.
    """
    parts = []

    # Add text part (skip if empty)
    if page_text.strip():
        parts.append(types.Part(text=page_text))

    # Add image part
    parts.append(
        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    )

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[types.Content(parts=parts)],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values


def main():
    # --- Setup ---
    PAGES_CACHE_DIR.mkdir(exist_ok=True)

    google_client = genai.Client(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = setup_pinecone_index(pc)

    # --- Open PDF ---
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"\nProcessing '{pdf_path.name}' — {total_pages} pages\n")

    vectors = []

    for i, page in enumerate(doc):
        page_num = i + 1
        print(f"  Page {page_num}/{total_pages}...", end=" ", flush=True)

        # Extract text
        page_text = page.get_text()

        # Render page to PNG image at 2x resolution for quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        image_path = PAGES_CACHE_DIR / f"page_{page_num}.png"
        pix.save(str(image_path))

        # Read image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Create multimodal embedding (text + image → single vector)
        embedding = embed_page(google_client, page_text, image_bytes)

        # Prepare Pinecone record
        vectors.append({
            "id": f"page-{page_num}",
            "values": embedding,
            "metadata": {
                "page_number": page_num,
                "text": page_text[:800],  # Store first 800 chars as snippet
                "image_path": str(image_path),
                "pdf_name": pdf_path.name,
            },
        })

        print(f"✓ embedded ({len(embedding)} dims)")

    doc.close()

    # --- Upsert all vectors to Pinecone ---
    print(f"\nUpserting {len(vectors)} vectors to Pinecone...")
    # Upsert in batches of 10
    batch_size = 10
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch)
    print("Done! All pages ingested into Pinecone.")
    print(f"Page images saved in: {PAGES_CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()
