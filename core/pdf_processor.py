"""
IntelliDoc AI - PDF Processing Module
Extracts text from PDF documents using PyPDF with comprehensive validation
for corrupted, encrypted, scanned, or empty files.
"""

import io
from pathlib import Path
from typing import Dict, Any, List, Union
from pypdf import PdfReader
from pypdf.errors import PdfReadError, FileNotDecryptedError


def process_pdf(pdf_source: Union[str, Path, io.BytesIO, bytes], filename: str = "document.pdf") -> Dict[str, Any]:
    """
    Validates and extracts page-by-page text from an uploaded or local PDF.

    Args:
        pdf_source: File path, BytesIO stream, or raw bytes.
        filename: Optional name of the PDF for metadata.

    Returns:
        Dict containing success status, page count, character count,
        page-by-page text dictionaries, or an error description.
    """
    try:
        # Normalize input to a stream or file path
        if isinstance(pdf_source, (str, Path)):
            stream = str(pdf_source)
            if not Path(stream).exists():
                return {
                    "success": False,
                    "filename": filename,
                    "error": f"PDF file not found at path: {pdf_source}",
                }
            reader = PdfReader(stream)
        elif isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
            reader = PdfReader(stream)
        elif isinstance(pdf_source, io.BytesIO):
            reader = PdfReader(pdf_source)
        elif hasattr(pdf_source, "read"):
            # Handle Streamlit UploadedFile
            stream = io.BytesIO(pdf_source.read())
            reader = PdfReader(stream)
        else:
            return {
                "success": False,
                "filename": filename,
                "error": "Unsupported PDF source format.",
            }

        # 1. Check encryption / password protection
        if reader.is_encrypted:
            return {
                "success": False,
                "filename": filename,
                "error": "This PDF is encrypted or password-protected. Please unlock or decrypt the document before uploading.",
            }

        page_count = len(reader.pages)
        if page_count == 0:
            return {
                "success": False,
                "filename": filename,
                "page_count": 0,
                "error": "The uploaded PDF contains 0 pages.",
            }

        pages_data: List[Dict[str, Any]] = []
        total_characters = 0

        # 2. Extract text page by page preserving page numbering
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            extracted = page.extract_text() or ""
            cleaned_text = extracted.strip()

            if cleaned_text:
                total_characters += len(cleaned_text)

            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
            })

        # 3. Detect scanned / image-only PDFs with zero extractable text
        if total_characters == 0:
            return {
                "success": False,
                "filename": filename,
                "page_count": page_count,
                "total_characters": 0,
                "pages": pages_data,
                "error": (
                    "No extractable digital text found in this PDF. "
                    "The document appears to be scanned or contains image-only pages. "
                    "Please upload a PDF containing selectable text."
                ),
            }

        return {
            "success": True,
            "filename": filename,
            "page_count": page_count,
            "total_characters": total_characters,
            "pages": pages_data,
            "error": None,
        }

    except FileNotDecryptedError:
        return {
            "success": False,
            "filename": filename,
            "error": "This PDF is password-protected and cannot be read without a password.",
        }
    except PdfReadError as e:
        return {
            "success": False,
            "filename": filename,
            "error": f"Failed to parse PDF document (corrupted or malformed file): {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "filename": filename,
            "error": f"Unexpected error while processing PDF: {str(e)}",
        }
