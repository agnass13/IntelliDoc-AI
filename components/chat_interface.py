"""
IntelliDoc AI - Chat Interface Component
Manages conversation rendering, streaming answers, and source citations.
"""

import streamlit as st
from core.rag_pipeline import RAGPipeline


def render_chat_interface(rag_pipeline: RAGPipeline):
    """
    Renders the chat stream, history, suggestion pills, and query input.
    """
    current_doc = st.session_state.get("current_doc")
    chat_history = st.session_state.setdefault("chat_history", [])

    # If no document is uploaded yet, display friendly guidance
    if not current_doc:
        st.info("👋 **Welcome to IntelliDoc AI!**\nPlease upload a PDF document in the left sidebar to start asking questions.", icon="ℹ️")

        with st.expander("💡 How It Works (RAG Architecture)", expanded=True):
            st.markdown(
                """
                1. **PDF Processing:** Digital text is extracted page by page using `PyPDF`.
                2. **Semantic Chunking:** Text is split into 1,000-character overlapping chunks preserving page metadata.
                3. **Local Embeddings:** Text chunks are transformed into dense vectors via `nomic-embed-text`.
                4. **Vector Search:** `ChromaDB` retrieves the top relevant excerpts based on cosine similarity.
                5. **Local LLM Synthesis:** `llama3.2:3b` generates factual answers strictly grounded in your document.
                """
            )
        return

    # Render previous conversation history
    for message in chat_history:
        role = message.get("role")
        content = message.get("content")
        sources = message.get("sources", [])

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
            st.markdown(content)

            # Display source citations if available
            if role == "assistant" and sources:
                with st.expander("📚 View Source Citations & Document Excerpts", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        page_num = src.get("page", "Unknown")
                        text_excerpt = src.get("text", "")
                        st.markdown(f"**Source #{idx} (Page {page_num})**")
                        st.caption(f"> {text_excerpt}")
                        st.divider()

    # Suggestion questions if conversation is new
    if not chat_history:
        st.markdown("##### 💡 Suggested Questions")
        col1, col2, col3 = st.columns(3)
        suggestions = [
            "What is the main topic of this document?",
            "Provide a concise executive summary.",
            "What are the key conclusions or takeaways?",
        ]

        clicked_prompt = None
        with col1:
            if st.button(suggestions[0], use_container_width=True):
                clicked_prompt = suggestions[0]
        with col2:
            if st.button(suggestions[1], use_container_width=True):
                clicked_prompt = suggestions[1]
        with col3:
            if st.button(suggestions[2], use_container_width=True):
                clicked_prompt = suggestions[2]

        if clicked_prompt:
            handle_user_query(clicked_prompt, rag_pipeline, current_doc)
            st.rerun()

    # Primary Chat Input
    user_query = st.chat_input("Ask any question regarding the uploaded PDF...")
    if user_query:
        handle_user_query(user_query, rag_pipeline, current_doc)
        st.rerun()


def handle_user_query(query: str, rag_pipeline: RAGPipeline, current_doc: dict):
    """
    Processes the user's query through the RAG pipeline, streams response,
    and appends to conversation history.
    """
    collection_name = current_doc.get("collection_name")
    if not collection_name:
        st.error("No active vector collection found for this document.")
        return

    # Add user message to history
    st.session_state["chat_history"].append({"role": "user", "content": query})

    # Render current user query
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Stream AI assistant response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Analyzing document context and synthesizing answer..."):
            stream_gen, retrieved_chunks = rag_pipeline.answer_question_stream(
                collection_name=collection_name,
                question=query,
            )
            response_container = st.empty()
            full_response = ""

            for chunk_token in stream_gen:
                full_response += chunk_token
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)

        # Append assistant message with source citations
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": full_response,
            "sources": retrieved_chunks,
        })
