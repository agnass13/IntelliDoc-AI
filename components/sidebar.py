"""
IntelliDoc AI - Sidebar Navigation & Controls
Handles file uploads, Ollama service diagnostics, and session cleanup.
"""

import streamlit as st
from config import settings
from core.pdf_processor import process_pdf
from core.text_splitter import split_document
from core.embeddings import check_required_models, generate_embeddings_batch
from core.vector_store import DocumentVectorStore
from utils.helpers import format_file_size


def render_sidebar(vector_store: DocumentVectorStore):
    """
    Renders the sidebar with document upload controls and system diagnostics.
    """
    with st.sidebar:
        # 1. Branding Header
        st.markdown("## 🧠 IntelliDoc AI")
        st.caption("Local RAG Document Question Answering")

        st.divider()

        # 2. Ollama Local Diagnostics Check
        st.markdown("### 🔌 AI Service Status")
        model_check = check_required_models()

        if model_check["all_ready"]:
            st.success("🟢 Local Ollama Active", icon="✅")
            with st.expander("Model Configuration", expanded=False):
                st.write(f"**LLM:** `{settings.LLM_MODEL}`")
                st.write(f"**Embeddings:** `{settings.EMBEDDING_MODEL}`")
                st.write(f"**Vector Store:** `ChromaDB`")
        elif model_check["ollama_running"]:
            st.warning("⚠️ Missing Models in Ollama", icon="⚠️")
            st.error(model_check["message"])
            with st.expander("Setup Instructions", expanded=True):
                st.code(
                    f"ollama pull {settings.LLM_MODEL}\nollama pull {settings.EMBEDDING_MODEL}",
                    language="powershell",
                )
        else:
            st.error("🔴 Ollama Offline", icon="❌")
            st.caption("Local Ollama service was not detected on port 11434.")
            with st.expander("How to Start Ollama", expanded=True):
                st.markdown(
                    "1. Install from [ollama.com](https://ollama.com)\n"
                    "2. Open PowerShell and run:\n"
                    "```powershell\nollama serve\n```\n"
                    "3. In another terminal, pull the models:\n"
                    "```powershell\n"
                    f"ollama pull {settings.LLM_MODEL}\n"
                    f"ollama pull {settings.EMBEDDING_MODEL}\n"
                    "```"
                )

        st.divider()

        # 3. PDF Upload Section
        st.markdown("### 📤 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document to begin",
            type=["pdf"],
            help="Selectable digital PDFs only. Password-protected files must be unlocked.",
        )

        # Trigger processing when a new PDF is uploaded
        if uploaded_file is not None:
            current_doc = st.session_state.get("current_doc")

            # Check if this is a newly uploaded file
            if not current_doc or current_doc.get("filename") != uploaded_file.name:
                with st.spinner(f"Processing and indexing {uploaded_file.name}..."):
                    # Step A: Validate and extract text
                    pdf_result = process_pdf(uploaded_file, filename=uploaded_file.name)

                    if not pdf_result["success"]:
                        st.error(pdf_result.get("error", "Failed to extract text from PDF."))
                    else:
                        # Step B: Split text into chunks
                        chunks = split_document(
                            pages_data=pdf_result["pages"],
                            filename=uploaded_file.name,
                        )

                        if not chunks:
                            st.error("Failed to generate text chunks from the document.")
                        else:
                            # Step C: Generate embeddings & store in ChromaDB
                            if not model_check["ollama_running"]:
                                st.error("Cannot index document: Local Ollama service is not running.")
                            else:
                                try:
                                    collection_name = vector_store.get_collection_name(uploaded_file.name)
                                    chunk_texts = [c["text"] for c in chunks]

                                    progress_bar = st.progress(0, text="Generating embeddings with nomic-embed-text...")
                                    embeddings = []
                                    total_chunks = len(chunk_texts)

                                    for i, text in enumerate(chunk_texts):
                                        vec = generate_embeddings_batch([text])
                                        if vec:
                                            embeddings.append(vec[0])
                                        progress_bar.progress(
                                            (i + 1) / total_chunks,
                                            text=f"Generating embeddings ({i+1}/{total_chunks})...",
                                        )

                                    progress_bar.empty()

                                    # Store in ChromaDB
                                    vector_store.store_chunks(
                                        collection_name=collection_name,
                                        chunks=chunks,
                                        embeddings=embeddings,
                                    )

                                    # Save to Streamlit session state
                                    st.session_state["current_doc"] = {
                                        "filename": uploaded_file.name,
                                        "page_count": pdf_result["page_count"],
                                        "total_characters": pdf_result["total_characters"],
                                        "chunk_count": len(chunks),
                                        "collection_name": collection_name,
                                        "file_size": format_file_size(uploaded_file.size),
                                    }
                                    st.session_state["chat_history"] = []
                                    st.success(f"Indexed {len(chunks)} chunks into vector store!")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error during embedding generation: {str(e)}")

        st.divider()

        # 4. Session & Chat Actions
        st.markdown("### ⚙️ Session Controls")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🧹 Clear Chat", use_container_width=True, help="Reset conversation history."):
                st.session_state["chat_history"] = []
                st.rerun()

        with col_btn2:
            if st.button("🗑️ Remove Doc", use_container_width=True, help="Remove document and purge vectors."):
                current_doc = st.session_state.get("current_doc")
                if current_doc and "collection_name" in current_doc:
                    vector_store.delete_document(current_doc["collection_name"])
                st.session_state["current_doc"] = None
                st.session_state["chat_history"] = []
                st.rerun()
