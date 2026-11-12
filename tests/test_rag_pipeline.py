"""
IntelliDoc AI - Vector Store & RAG Pipeline Unit Tests
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.vector_store import DocumentVectorStore
from core.embeddings import check_ollama_status
from core.rag_pipeline import RAGPipeline


def test_vector_store_operations():
    """Tests ChromaDB collection isolation, chunk storage, and cosine similarity query."""
    vs = DocumentVectorStore()
    col_name = vs.get_collection_name("sample_ai_paper.pdf")

    # Sample test chunks and artificial embeddings
    chunks = [
        {
            "id": "c1",
            "text": "Neural networks learn representations from high-dimensional data.",
            "metadata": {"source": "sample_ai_paper.pdf", "page": 1, "chunk_id": 1},
        },
        {
            "id": "c2",
            "text": "SQLite is a lightweight embedded database engine stored on disk.",
            "metadata": {"source": "sample_ai_paper.pdf", "page": 2, "chunk_id": 2},
        },
    ]

    # Dummy 4-dimensional normalized vectors for testing
    embeddings = [
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.6],
    ]

    # Store
    success = vs.store_chunks(col_name, chunks, embeddings)
    assert success is True

    # Search with a query embedding close to c1
    query_vec = [0.85, 0.15, 0.0, 0.0]
    results = vs.search_similar(col_name, query_vec, top_k=1)

    assert len(results) == 1
    assert "Neural networks" in results[0]["text"]
    assert results[0]["page"] == 1

    # Cleanup
    vs.delete_document(col_name)
    print("✓ test_vector_store_operations passed")


def test_ollama_offline_graceful_handling():
    """Ensures that when Ollama is offline, the status check returns False without raising unhandled exceptions."""
    status = check_ollama_status("http://127.0.0.1:9999")  # Non-existent port
    assert status["available"] is False
    assert status["error"] is not None
    print("✓ test_ollama_offline_graceful_handling passed")


if __name__ == "__main__":
    print("--- Running Vector Store & RAG Tests ---")
    test_vector_store_operations()
    test_ollama_offline_graceful_handling()
    print("All Vector Store & RAG tests passed successfully!")
