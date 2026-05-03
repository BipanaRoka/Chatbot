"""
ingest.py
─────────
Updated ingestion pipeline that:
  1. Loads PDFs from the data/pdfs/ folder  (or falls back to the .txt KB)
  2. Chunks them with RecursiveCharacterTextSplitter
  3. Embeds chunks with sentence-transformers
  4. Stores vectors in Qdrant

Usage:
    python ingest.py                          # load PDFs from data/pdfs/
    python ingest.py --source data/pdfs/      # explicit folder
    python ingest.py --source data/report.pdf # single file
    python ingest.py --loader pymupdf         # choose loader
    python ingest.py --chunk-size 600 --overlap 100
    python ingest.py --stats                  # show chunk stats, no upload
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from utils.pdf_loader import load_and_chunk, chunk_stats, inspect_chunks

load_dotenv()

# ── Config from .env ──────────────────────────────────────────────────────────
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION_NAME", "farm_plant_knowledge")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL       = os.getenv("QDRANT_URL")       # set for Qdrant Cloud
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")   # set for Qdrant Cloud
DEFAULT_PDF_DIR  = "data/pdfs"


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("   ✅ Embedding model ready.\n")
    return emb


def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        print(f"☁️  Qdrant Cloud: {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"🐳 Local Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_or_reset_collection(client: QdrantClient, vector_size: int = 384) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"   ♻️  Collection '{COLLECTION_NAME}' exists — recreating.")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"   ✅ Collection '{COLLECTION_NAME}' created.")


def ingest(
    source: str,
    loader_type: str = "pypdf",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    stats_only: bool = False,
) -> None:
    print("\n" + "=" * 55)
    print("  FarmWise RAG — Ingestion Pipeline")
    print("=" * 55 + "\n")

    # Step 1: Load & Chunk PDFs
    print(f"Source : {source}")
    print(f"Loader : {loader_type} | chunk_size={chunk_size} | overlap={chunk_overlap}\n")

    chunks = load_and_chunk(
        path=source,
        loader_type=loader_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=True,
    )

    if not chunks:
        print("No chunks produced. Check that PDFs exist in the source path.")
        return

    # Step 2: Show stats
    stats = chunk_stats(chunks)
    print("Chunk Statistics:")
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Unique files    : {stats['unique_files']}")
    print(f"  Files           : {', '.join(stats['files'])}")
    print(f"  Avg chunk size  : {stats['avg_chunk_length']} chars")
    print(f"  Min / Max       : {stats['min_chunk_length']} / {stats['max_chunk_length']} chars\n")

    if stats_only:
        print("--stats mode: skipping embedding & upload.")
        inspect_chunks(chunks, n=3)
        return

    # Step 3: Embed & Upload to Qdrant
    embeddings = get_embeddings()
    client     = get_qdrant_client()

    create_or_reset_collection(client, vector_size=384)

    print(f"\nEmbedding {len(chunks)} chunks and uploading to Qdrant...")

    qdrant_url = QDRANT_URL if QDRANT_URL else f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    print(f"  {len(chunks)} chunks stored in '{COLLECTION_NAME}'.")
    print("\nDone! Run the app with:  streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FarmWise PDF Ingestion Pipeline")
    parser.add_argument(
        "--source", default=DEFAULT_PDF_DIR,
        help="Path to a PDF file or folder of PDFs (default: data/pdfs/)",
    )
    parser.add_argument(
        "--loader", default="pypdf",
        choices=["pypdf", "pymupdf", "pdfminer"],
        help="PDF loader backend (default: pypdf)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--overlap", type=int, default=80,
        help="Overlap between chunks (default: 80)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show chunk stats only — skip embedding & upload",
    )
    args = parser.parse_args()

    ingest(
        source=args.source,
        loader_type=args.loader,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        stats_only=args.stats,
    )