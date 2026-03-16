"""
app.py — Streamlit "Manual Assistant" UI.

Matches the design from the screenshots:
- Header with document icon
- Chat messages with formatted answers
- Page thumbnail thumbnails (Page 17, Page 39, etc.)
- Green match percentage badges
- Expandable "N sources" section
- "Ask about the manual..." chat input
"""

import streamlit as st
from pathlib import Path
from rag import query_rag

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manual Assistant",
    page_icon="📄",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container width */
    .main .block-container { max-width: 800px; padding-top: 1rem; }

    /* Header */
    .manual-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 0 16px 0;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 24px;
    }
    .manual-header-icon {
        background: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 6px 8px;
        font-size: 20px;
        line-height: 1;
    }
    .manual-header-title {
        font-size: 20px;
        font-weight: 600;
        color: #202124;
    }

    /* Source card */
    .source-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
    }
    .source-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .page-label {
        font-size: 13px;
        font-weight: 600;
        color: #1a73e8;
    }
    .match-badge {
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 600;
    }
    .source-snippet {
        font-size: 13px;
        color: #5f6368;
        line-height: 1.5;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }

    /* Thumbnail label */
    .thumb-label {
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        color: #5f6368;
        margin-top: 4px;
        background: rgba(0,0,0,0.55);
        color: white;
        border-radius: 0 0 6px 6px;
        padding: 2px 0;
    }

    /* Chat input area */
    .stChatInput > div { border-radius: 24px !important; }

    /* Assistant message bubble */
    [data-testid="stChatMessage"] { padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="manual-header">
        <div class="manual-header-icon">📄</div>
        <div class="manual-header-title">Manual Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources}

# ─── Render existing messages ────────────────────────────────────────────────
def render_sources(sources: list[dict]):
    """Render page thumbnails + expandable sources section."""
    if not sources:
        return

    # Top 3 thumbnails row
    top_sources = sources[:3]
    cols = st.columns(len(top_sources))
    for col, src in zip(cols, top_sources):
        img_path = Path(src["image_path"])
        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.markdown(
                    f"<div style='background:#f1f3f4;border-radius:6px;height:120px;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"color:#9aa0a6;font-size:12px'>No image</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div class='thumb-label'>Page {src['page_number']}</div>",
                unsafe_allow_html=True,
            )

    # Expandable sources list
    with st.expander(f"▶  {len(sources)} sources"):
        for src in sources:
            score_color = "#2e7d32" if src["score"] >= 70 else "#f57f17"
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-card-header">
                        <span class="page-label">Page {src['page_number']}</span>
                        <span class="match-badge" style="color:{score_color}">
                            {src['score']}% match
                        </span>
                    </div>
                    <div class="source-snippet">{src['text_snippet']}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ─── Chat input ──────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about the manual..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching the manual..."):
            answer, sources = query_rag(question, top_k=5)

        st.markdown(answer)
        render_sources(sources)

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
