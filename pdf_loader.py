"""
utils/pdf_loader.py
───────────────────
Handles everything related to loading PDF files and splitting them
into chunks ready for embedding into Qdrant.

Supports:
  • Single PDF file
  • Entire folder of PDFs  (recursive optional)
  • Scanned PDFs via OCR   (requires pytesseract + poppler)

Pipeline:
  load_pdfs()  →  List[Document]
  chunk_documents()  →  List[Document]
  load_and_chunk()   →  List[Document]   (convenience wrapper)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,           # fast, text-based PDFs
    PyMuPDFLoader,         # better metadata extraction
    PDFMinerLoader,        # best for complex layouts
    PyPDFDirectoryLoader,  # loads whole folders of PDFs
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


#PDF Loading

def load_single_pdf(
    file_path: str,
    loader_type: str = "pypdf",
) -> List[Document]:
    """
    Load a single PDF file and return a list of Document objects.

    Args:
        file_path   : Absolute or relative path to the PDF file.
        loader_type : Which loader to use — 'pypdf' | 'pymupdf' | 'pdfminer'
                      • pypdf    → fast, good for standard text PDFs
                      • pymupdf  → faster + richer metadata (page size, fonts)
                      • pdfminer → best for complex multi-column layouts

    Returns:
        List[Document] — one Document per page, each with metadata:
            source, page, total_pages, file_name
    """
    file_path = str(Path(file_path).resolve())
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Loading PDF [{loader_type}]: {file_path}")

    loaders = {
        "pypdf":    PyPDFLoader,
        "pymupdf":  PyMuPDFLoader,
        "pdfminer": PDFMinerLoader,
    }
    loader_cls = loaders.get(loader_type, PyPDFLoader)
    loader = loader_cls(file_path)
    docs = loader.load()

    # Enrich metadata
    file_name = Path(file_path).name
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source":      file_path,
            "file_name":   file_name,
            "page":        doc.metadata.get("page", i),
            "total_pages": len(docs),
        })

    logger.info(f"  → Loaded {len(docs)} pages from '{file_name}'")
    return docs


def load_pdf_folder(
    folder_path: str,
    glob: str = "**/*.pdf",
    loader_type: str = "pypdf",
) -> List[Document]:
    """
    Load all PDF files inside a folder (recursive by default).

    Args:
        folder_path : Path to the folder containing PDFs.
        glob        : File pattern — '**/*.pdf' loads recursively,
                      '*.pdf' loads only the top level.
        loader_type : 'pypdf' | 'pymupdf' | 'pdfminer'

    Returns:
        List[Document] — all pages from all PDFs, with metadata.
    """
    folder_path = str(Path(folder_path).resolve())
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    pdf_files = list(Path(folder_path).glob(glob))
    if not pdf_files:
        logger.warning(f"No PDF files found in: {folder_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in '{folder_path}'")

    all_docs: List[Document] = []
    for pdf_path in sorted(pdf_files):
        try:
            docs = load_single_pdf(str(pdf_path), loader_type=loader_type)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"  ✗ Failed to load '{pdf_path.name}': {e}")

    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs


def load_pdfs(
    path: str,
    loader_type: str = "pypdf",
    glob: str = "**/*.pdf",
) -> List[Document]:
    """
    Auto-detects whether `path` is a file or folder and loads accordingly.

    Args:
        path        : Path to a single PDF or a folder of PDFs.
        loader_type : 'pypdf' | 'pymupdf' | 'pdfminer'
        glob        : Used only when path is a folder.

    Returns:
        List[Document]
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return load_single_pdf(str(p), loader_type=loader_type)
    elif p.is_dir():
        return load_pdf_folder(str(p), glob=glob, loader_type=loader_type)
    else:
        raise ValueError(f"Path must be a .pdf file or a directory: {path}")


# 2. Chunking / Text Splitting

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: Optional[List[str]] = None,
    add_chunk_metadata: bool = True,
) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks for embedding.

    Why overlap?
        Overlap ensures that sentences or ideas that straddle a chunk
        boundary still appear fully in at least one chunk, improving
        retrieval accuracy.

    Args:
        docs               : Raw Document list from load_pdfs().
        chunk_size         : Max characters per chunk.
        chunk_overlap      : Characters shared between adjacent chunks.
        separators         : Custom split hierarchy. Defaults to a set
                             that respects paragraphs → sentences → words.
        add_chunk_metadata : If True, adds chunk_index and chunk_total
                             to each chunk's metadata.

    Returns:
        List[Document] — chunked documents with enriched metadata.
    """
    if not docs:
        logger.warning("chunk_documents() received an empty document list.")
        return []

    if separators is None:
        # Priority order: try to split on these boundaries first
        separators = [
            "\n\n",    # paragraph break
            "\n",      # line break
            ". ",      # sentence end
            ", ",      # clause break
            " ",       # word boundary
            "",        # character (last resort)
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    if add_chunk_metadata:
        # Group by source file so chunk indices are per-document
        from collections import defaultdict
        source_chunks: dict = defaultdict(list)
        for chunk in chunks:
            source_chunks[chunk.metadata.get("source", "unknown")].append(chunk)

        for source, source_chunk_list in source_chunks.items():
            total = len(source_chunk_list)
            for idx, chunk in enumerate(source_chunk_list):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["chunk_total"] = total

    logger.info(
        f"Chunking complete: {len(docs)} pages → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# 3. Convenience Wrapper

def load_and_chunk(
    path: str,
    loader_type: str = "pypdf",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    glob: str = "**/*.pdf",
    verbose: bool = True,
) -> List[Document]:
    """
    One-shot function: load PDFs + chunk them.

    Args:
        path         : Single PDF path or folder path.
        loader_type  : 'pypdf' | 'pymupdf' | 'pdfminer'
        chunk_size   : Characters per chunk.
        chunk_overlap: Overlap between chunks.
        glob         : Folder glob pattern.
        verbose      : Print a preview of the first chunk if True.

    Returns:
        List[Document] — chunked, metadata-enriched documents.
    """
    docs = load_pdfs(path, loader_type=loader_type, glob=glob)

    if not docs:
        logger.warning("No documents loaded. Check your PDF path.")
        return []

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if verbose and chunks:
        print("\n" + "=" * 60)
        print("CHUNK PREVIEW (first chunk)")
        print("=" * 60)
        first = chunks[0]
        print(f"File     : {first.metadata.get('file_name', 'N/A')}")
        print(f"Page     : {first.metadata.get('page', 'N/A')}")
        print(f"Chunk    : {first.metadata.get('chunk_index', 0) + 1} / "
              f"{first.metadata.get('chunk_total', '?')}")
        print(f"Length   : {len(first.page_content)} characters")
        print("-" * 60)
        print(first.page_content[:400] + ("..." if len(first.page_content) > 400 else ""))
        print("=" * 60 + "\n") 

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diagnostic / Inspection Utilities
# ─────────────────────────────────────────────────────────────────────────────

def inspect_chunks(chunks: List[Document], n: int = 5) -> None:
    """Pretty-print the first `n` chunks for debugging."""
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks[:n]):
        meta = chunk.metadata
        print(f"\n[Chunk {i+1}]")
        print(f"  File    : {meta.get('file_name', 'N/A')}")
        print(f"  Page    : {meta.get('page', 'N/A')}")
        print(f"  Length  : {len(chunk.page_content)} chars")
        print(f"  Preview : {chunk.page_content[:150].strip()}...")
    print(f"\n{'='*60}\n")


def chunk_stats(chunks: List[Document]) -> dict:
    """Return basic statistics about the chunk list."""
    if not chunks:
        return {}
    lengths = [len(c.page_content) for c in chunks]
    sources = {c.metadata.get("file_name", "unknown") for c in chunks}
    return {
        "total_chunks": len(chunks),
        "unique_files": len(sources),
        "files": sorted(sources),
        "avg_chunk_length": round(sum(lengths) / len(lengths)),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }