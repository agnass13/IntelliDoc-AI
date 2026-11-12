"""
IntelliDoc AI - Text Splitting Module
Splits extracted document text into manageable chunks while preserving
crucial metadata such as page numbers and document filenames.
"""

from typing import List, Dict, Any
from config import settings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_document(
    pages_data: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page text into semantic chunks with overlap.

    Args:
        pages_data: List of page dictionaries [{"page_number": 1, "text": "..."}]
        filename: Name of the source document
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between consecutive chunks

    Returns:
        List of chunk dictionaries with text and metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Dict[str, Any]] = []
    chunk_id = 1

    for page in pages_data:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "").strip()

        if not page_text:
            continue

        raw_splits = text_splitter.split_text(page_text)

        for split_text in raw_splits:
            cleaned = split_text.strip()
            if not cleaned:
                continue

            chunks.append({
                "id": f"{filename}_p{page_num}_c{chunk_id}",
                "text": cleaned,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "length": len(cleaned),
                },
            })
            chunk_id += 1

    return chunks
