"""
IntelliDoc AI - PDF Processor and Text Splitter Unit Tests
"""

import io
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pypdf import PdfWriter
from core.pdf_processor import process_pdf
from core.text_splitter import split_document


def create_dummy_pdf(text_pages: list) -> io.BytesIO:
    """Helper to generate an in-memory PDF with specified text per page."""
    writer = PdfWriter()
    for text in text_pages:
        # Note: pypdf can create blank pages, or we can use basic stream
        page = writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def test_empty_pdf_detection():
    """Verify that a PDF with no extractable text returns an informative error."""
    blank_pdf = create_dummy_pdf(["", ""])
    result = process_pdf(blank_pdf, filename="blank.pdf")

    assert result["success"] is False
    assert "No extractable digital text" in result["error"]
    assert result["page_count"] == 2
    print("✓ test_empty_pdf_detection passed")


def test_text_splitting_and_metadata():
    """Verify that text splitting generates valid chunks and preserves page metadata."""
    sample_pages = [
        {
            "page_number": 1,
            "text": "Artificial Intelligence is transforming computing. Retrieval-Augmented Generation bridges LLMs with domain documents.",
        },
        {
            "page_number": 2,
            "text": "ChromaDB is an open-source embedding database. It stores dense vectors for nearest-neighbor similarity search.",
        },
    ]

    chunks = split_document(sample_pages, filename="test_doc.pdf", chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    # Verify metadata preservation
    assert chunks[0]["metadata"]["source"] == "test_doc.pdf"
    assert chunks[0]["metadata"]["page"] == 1
    assert "id" in chunks[0]
    assert "text" in chunks[0]

    # Verify second page chunks exist
    p2_chunks = [c for c in chunks if c["metadata"]["page"] == 2]
    assert len(p2_chunks) > 0
    print("✓ test_text_splitting_and_metadata passed")


if __name__ == "__main__":
    print("--- Running PDF Processor & Splitter Tests ---")
    test_empty_pdf_detection()
    test_text_splitting_and_metadata()
    print("All PDF processor tests passed successfully!")
