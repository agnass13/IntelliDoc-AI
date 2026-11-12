"""
IntelliDoc AI - Utility Helpers
Provides formatting, size conversion, and string sanitization utilities.
"""

def format_file_size(size_bytes: int) -> str:
    """Formats raw bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def truncate_text(text: str, max_chars: int = 150) -> str:
    """Truncates text with ellipsis for preview badges."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."
