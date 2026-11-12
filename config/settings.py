"""
IntelliDoc AI - Application Settings & Configuration
Loads environment variables and sets sensible defaults for RAG pipeline,
Ollama service connection, and document storage.
"""

import os
from pathlib import Path

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if python-dotenv is available
ENV_PATH = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    # Fallback to standard os.environ if dotenv not yet installed
    pass

# ==========================================
# Ollama Local Service Configuration
# ==========================================
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ==========================================
# RAG Chunking & Retrieval Parameters
# ==========================================
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "4"))

# ==========================================
# Storage Directories
# ==========================================
DATA_DIR: Path = BASE_DIR / "data"
VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"

def ensure_directories():
    """Ensure data and vector_db directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Run directory creation on import
ensure_directories()
