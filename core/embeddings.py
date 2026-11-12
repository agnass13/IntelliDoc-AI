"""
IntelliDoc AI - Ollama Embedding Module
Generates dense numerical vector representations of document chunks
using the locally running Ollama nomic-embed-text model.
"""

from typing import List, Dict, Any, Optional
import requests
from config import settings


def check_ollama_status(base_url: str = settings.OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Checks whether the local Ollama daemon is reachable and lists available models.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return {
                "available": True,
                "models": models,
                "error": None,
            }
        return {
            "available": False,
            "models": [],
            "error": f"Ollama returned HTTP status {response.status_code}",
        }
    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "models": [],
            "error": "Ollama service is not running. Please start Ollama (e.g., 'ollama serve' in PowerShell).",
        }
    except Exception as e:
        return {
            "available": False,
            "models": [],
            "error": f"Error connecting to Ollama: {str(e)}",
        }


def check_required_models(
    embedding_model: str = settings.EMBEDDING_MODEL,
    llm_model: str = settings.LLM_MODEL,
    base_url: str = settings.OLLAMA_BASE_URL,
) -> Dict[str, Any]:
    """
    Verifies that both required models are pulled and ready in Ollama.
    """
    status = check_ollama_status(base_url)
    if not status["available"]:
        return {
            "all_ready": False,
            "ollama_running": False,
            "missing_models": [embedding_model, llm_model],
            "message": status["error"],
        }

    installed = status["models"]

    def is_present(target: str) -> bool:
        # Matches e.g. "nomic-embed-text:latest" or "nomic-embed-text"
        base_target = target.split(":")[0]
        return any(base_target in m for m in installed)

    missing = []
    if not is_present(embedding_model):
        missing.append(embedding_model)
    if not is_present(llm_model):
        missing.append(llm_model)

    return {
        "all_ready": len(missing) == 0,
        "ollama_running": True,
        "installed_models": installed,
        "missing_models": missing,
        "message": (
            "All required models are installed and ready."
            if len(missing) == 0
            else f"Missing required models: {', '.join(missing)}. Run: `ollama pull {'` and `ollama pull '.join(missing)}`"
        ),
    }


def generate_embedding(
    text: str,
    model: str = settings.EMBEDDING_MODEL,
    base_url: str = settings.OLLAMA_BASE_URL,
) -> Optional[List[float]]:
    """
    Generates a single dense vector embedding for a given text snippet.
    """
    if not text or not text.strip():
        return None

    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", [])
        else:
            raise RuntimeError(
                f"Ollama embedding API failed with status {response.status_code}: {response.text}"
            )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollama service is unreachable at http://localhost:11434. "
            "Please ensure Ollama is running."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")


def generate_embeddings_batch(
    texts: List[str],
    model: str = settings.EMBEDDING_MODEL,
    base_url: str = settings.OLLAMA_BASE_URL,
) -> List[List[float]]:
    """
    Generates embeddings for a batch of text chunks sequentially.
    """
    embeddings = []
    for text in texts:
        vec = generate_embedding(text, model=model, base_url=base_url)
        if vec:
            embeddings.append(vec)
    return embeddings
