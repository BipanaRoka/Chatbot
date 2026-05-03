"""
test_pdf_loader.py
──────────────────
Quick script to test PDF loading and chunking WITHOUT needing
Qdrant or any API key.

Run it to verify your PDF files load and chunk correctly before
running the full ingestion pipeline.

Usage:
    python test_pdf_loader.py                     # test data/pdfs/ folder
    python test_pdf_loader.py --source myfile.pdf # test a specific file
"""

import argparse
from utils.pdf_loader import load_pdfs, chunk_documents, chunk_stats, inspect_chunks


def run_test(source: str, loader_type: str, chunk_size: int, overlap: int):
    print("\n" + "=" * 55)
    print("  PDF Loader Test")
    print("=" * 55)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print(f"\nStep 1: Loading PDFs from '{source}' using [{loader_type}]...")
    try:
        docs = load_pdfs(source, loader_type=loader_type)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"\n  ERROR: {e}")
        print("  Make sure your PDF files are inside the  data/pdfs/  folder.")
        return

    if not docs:
        print("\n  No pages loaded. Is the folder empty?")
        return

    print(f"\n  Loaded {len(docs)} total pages.")
    print("\n  Page metadata preview (first 3 pages):")
    for doc in docs[:3]:
        m = doc.metadata
        print(f"    File: {m.get('file_name','?')} | "
              f"Page {m.get('page','?')}/{m.get('total_pages','?')} | "
              f"Content length: {len(doc.page_content)} chars")

    # ── 2. Chunk ───────────────────────────────────────────────────────────
    print(f"\nStep 2: Chunking with size={chunk_size}, overlap={overlap}...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)

    # ── 3. Stats ───────────────────────────────────────────────────────────
    print("\nStep 3: Chunk Statistics")
    stats = chunk_stats(chunks)
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Unique files    : {stats['unique_files']}")
    print(f"  Avg chunk size  : {stats['avg_chunk_length']} chars")
    print(f"  Min / Max size  : {stats['min_chunk_length']} / {stats['max_chunk_length']} chars")

    # ── 4. Inspect ─────────────────────────────────────────────────────────
    print("\nStep 4: First 3 chunks preview")
    inspect_chunks(chunks, n=3)

    print("All tests passed! Your PDFs are ready for ingestion.")
    print("Run:  python ingest.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PDF loading and chunking")
    parser.add_argument("--source", default="data/pdfs", help="PDF file or folder")
    parser.add_argument("--loader", default="pypdf",
                        choices=["pypdf", "pymupdf", "pdfminer"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    run_test(args.source, args.loader, args.chunk_size, args.overlap)