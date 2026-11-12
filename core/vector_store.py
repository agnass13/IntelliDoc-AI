"""
IntelliDoc AI - ChromaDB Vector Store Module
Manages persistent local vector database storage, semantic similarity search,
and document-level isolation to prevent cross-contamination.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import settings


class DocumentVectorStore:
    """
    Manages local ChromaDB collections for document retrieval.
    """

    def __init__(self, persist_directory: str = str(settings.VECTOR_DB_DIR)):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def get_collection_name(filename: str) -> str:
        """
        Derives a safe, isolated collection name for ChromaDB.
        ChromaDB collection names must be 3-63 characters, alphanumeric, underscores, or hyphens.
        """
        # Clean alphanumeric representation
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)[:20]
        # Append hash to ensure uniqueness and document isolation
        name_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]
        return f"doc_{clean_name}_{name_hash}"

    def store_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> bool:
        """
        Stores chunk text, metadata, and numerical embeddings in ChromaDB.
        """
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must be non-empty and have matching lengths.")

        # Re-create collection cleanly to avoid stale data
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # ChromaDB batch insertion
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return True

    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = settings.TOP_K_CHUNKS,
    ) -> List[Dict[str, Any]]:
        """
        Searches the collection for chunks with the highest semantic similarity.
        """
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            return []

        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "Document"),
                })

        return retrieved

    def delete_document(self, collection_name: str) -> bool:
        """
        Removes the specified collection from ChromaDB.
        """
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """
        Clears all stored collections in the vector database.
        """
        try:
            collections = self.client.list_collections()
            for col in collections:
                # Handle both collection object and string name
                name = col.name if hasattr(col, "name") else str(col)
                self.client.delete_collection(name=name)
            return True
        except Exception:
            return False
