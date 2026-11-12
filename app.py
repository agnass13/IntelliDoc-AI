"""
IntelliDoc AI - Main Streamlit Application
A modern, local RAG question-answering system using Ollama, ChromaDB, and PyPDF.
"""

import streamlit as st
from config import settings
from core.vector_store import DocumentVectorStore
from core.rag_pipeline import RAGPipeline
from components.sidebar import render_sidebar
from components.document_info import render_document_info
from components.chat_interface import render_chat_interface

# Page Configuration
st.set_page_config(
    page_title="IntelliDoc AI - Local PDF Question Answering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Style CSS
st.markdown(
    """
    <style>
    /* Global Container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }

    /* Custom Header Tagline */
    .app-header {
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vector_store() -> DocumentVectorStore:
    """Instantiate and cache the ChromaDB vector database manager."""
    return DocumentVectorStore(persist_directory=str(settings.VECTOR_DB_DIR))


@st.cache_resource
def get_rag_pipeline() -> RAGPipeline:
    """Instantiate and cache the RAG pipeline."""
    vs = get_vector_store()
    return RAGPipeline(
        vector_store=vs,
        llm_model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        top_k=settings.TOP_K_CHUNKS,
    )


def main():
    # Initialize core services
    vector_store = get_vector_store()
    rag_pipeline = get_rag_pipeline()

    # Application Header
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">🧠 IntelliDoc AI</h1>
            <p class="app-subtitle">Local Retrieval-Augmented Generation (RAG) Document Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Sidebar (Upload, Diagnostics, Session Management)
    render_sidebar(vector_store)

    # Render Document Metadata Info (If a file is indexed)
    render_document_info()

    # Render Chat Stream & Interaction Panel
    render_chat_interface(rag_pipeline)


if __name__ == "__main__":
    main()
