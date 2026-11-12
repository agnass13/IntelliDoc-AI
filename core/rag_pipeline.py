"""
IntelliDoc AI - RAG Question-Answering Pipeline
Connects vector retrieval, context aggregation, and local Ollama LLM generation
with strict grounding prompts to prevent hallucination.
"""

from typing import List, Dict, Any, Generator, Tuple
import json
import requests
from config import settings
from core.embeddings import generate_embedding
from core.vector_store import DocumentVectorStore


SYSTEM_PROMPT = """You are IntelliDoc AI, an expert academic and technical document research assistant.
Your task is to answer the user's question accurately and objectively, relying EXCLUSIVELY on the provided document context excerpts below.

CRITICAL INSTRUCTIONS:
1. ONLY use information directly stated in the context below. Do NOT use outside general knowledge or make assumptions.
2. If the answer cannot be found in or deduced from the provided context, clearly state: "The provided document does not contain sufficient information to answer this question."
3. Cite the relevant source page numbers (e.g., "[Page 2]") when referencing specific facts or statements.
4. Keep your answer concise, well-structured, and factual.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""


class RAGPipeline:
    """
    Coordinates Retrieval-Augmented Generation using ChromaDB and local Ollama models.
    """

    def __init__(
        self,
        vector_store: DocumentVectorStore,
        llm_model: str = settings.LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        top_k: int = settings.TOP_K_CHUNKS,
    ):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.base_url = base_url
        self.top_k = top_k

    def retrieve_context(self, collection_name: str, question: str) -> List[Dict[str, Any]]:
        """
        Embeds the question and retrieves the top-k most relevant chunks.
        """
        query_embedding = generate_embedding(
            text=question,
            base_url=self.base_url,
        )
        if not query_embedding:
            return []

        chunks = self.vector_store.search_similar(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=self.top_k,
        )
        return chunks

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into clean numbered context blocks with page references.
        """
        if not retrieved_chunks:
            return "No relevant context found."

        formatted_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            page_num = chunk.get("page", "Unknown")
            text = chunk.get("text", "").strip()
            formatted_blocks.append(f"[Excerpt {i} | Page {page_num}]\n{text}")

        return "\n\n".join(formatted_blocks)

    def answer_question_stream(
        self,
        collection_name: str,
        question: str,
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Retrieves context and yields answer tokens in real-time for Streamlit chat.
        Returns a tuple of (token_generator, retrieved_chunks).
        """
        retrieved_chunks = self.retrieve_context(collection_name, question)

        if not retrieved_chunks:
            def empty_generator():
                yield "I could not find any relevant information regarding your question in the uploaded document."
            return empty_generator(), []

        context_str = self.format_context(retrieved_chunks)
        prompt_system = SYSTEM_PROMPT.format(context=context_str)

        def stream_generator():
            try:
                payload = {
                    "model": self.llm_model,
                    "prompt": question,
                    "system": prompt_system,
                    "stream": True,
                    "options": {
                        "temperature": settings.LLM_TEMPERATURE,
                    },
                }

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=60,
                )

                if response.status_code != 200:
                    yield f"Error from Ollama ({response.status_code}): {response.text}"
                    return

                for line in response.iter_lines():
                    if line:
                        chunk_json = json.loads(line.decode("utf-8"))
                        token = chunk_json.get("response", "")
                        yield token
                        if chunk_json.get("done", False):
                            break

            except requests.exceptions.ConnectionError:
                yield "Error: Could not connect to the local Ollama service. Make sure 'ollama serve' is running."
            except Exception as e:
                yield f"An error occurred while generating the answer: {str(e)}"

        return stream_generator(), retrieved_chunks

    def answer_question(
        self,
        collection_name: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Non-streaming answer generation (useful for programmatic testing).
        """
        generator, chunks = self.answer_question_stream(collection_name, question)
        full_text = "".join(list(generator))
        return {
            "answer": full_text,
            "sources": chunks,
        }
