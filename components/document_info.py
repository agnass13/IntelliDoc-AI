"""
IntelliDoc AI - Document Information Component
Renders metadata and statistics about the currently indexed PDF document.
"""

import streamlit as st
from utils.helpers import format_file_size


def render_document_info():
    """
    Renders top-level document stats and active vector index metrics.
    """
    doc_meta = st.session_state.get("current_doc")
    if not doc_meta:
        return

    st.markdown("### 📄 Document Intelligence Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Filename",
            value=doc_meta.get("filename", "N/A"),
            help="Name of the currently indexed PDF document.",
        )

    with col2:
        st.metric(
            label="Total Pages",
            value=f"{doc_meta.get('page_count', 0)} pages",
            help="Total number of digital pages parsed by PyPDF.",
        )

    with col3:
        st.metric(
            label="Extracted Text",
            value=f"{doc_meta.get('total_characters', 0):,} chars",
            help="Total character count extracted from the document.",
        )

    with col4:
        st.metric(
            label="Vector Chunks",
            value=f"{doc_meta.get('chunk_count', 0)} chunks",
            help="Number of semantic chunks stored in ChromaDB.",
        )

    st.divider()
