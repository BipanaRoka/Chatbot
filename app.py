# """
# app.py - Agri-RAG Chatbot
# Run with: streamlit run app.py
# """

# import streamlit as st
# from utils.rag_chain import build_rag_chain

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Agri-RAG Chatbot",
#     page_icon="🌾",
#     layout="centered",
# )

# # ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

# html, body, [class*="css"] {
#     font-family: 'IBM Plex Sans', sans-serif;
#     background: #f9fdf5;
# }

# /* ── Header ── */
# .site-header {
#     text-align: center;
#     padding: 2rem 1rem 1.4rem;
#     border-bottom: 1.5px solid #d4e8bc;
#     margin-bottom: 1.8rem;
# }
# .site-title {
#     font-family: 'Lora', serif;
#     font-size: 2rem;
#     font-weight: 700;
#     color: #1e3d0c;
#     margin: 0 0 0.35rem;
# }
# .site-title span { color: #4e9a1a; }
# .site-sub {
#     font-size: 0.88rem;
#     color: #607850;
#     margin: 0;
#     line-height: 1.5;
# }
# .site-sub b { color: #2e6010; }

# /* ── Q&A blocks ── */
# .qa-block {
#     background: #ffffff;
#     border: 1px solid #daecc4;
#     border-radius: 10px;
#     margin-bottom: 1.2rem;
#     overflow: hidden;
#     box-shadow: 0 1px 4px rgba(0,0,0,0.05);
# }
# .qa-question {
#     background: #f0f9e4;
#     border-bottom: 1px solid #daecc4;
#     padding: 0.75rem 1rem;
#     display: flex;
#     gap: 0.6rem;
#     align-items: flex-start;
# }
# .qa-question .icon {
#     font-size: 0.8rem;
#     font-weight: 700;
#     color: #4e9a1a;
#     background: #d6f0b2;
#     border-radius: 4px;
#     padding: 2px 6px;
#     margin-top: 1px;
#     white-space: nowrap;
#     flex-shrink: 0;
# }
# .qa-question .text {
#     font-size: 0.92rem;
#     color: #1e3d0c;
#     font-weight: 500;
#     line-height: 1.5;
# }
# .qa-answer {
#     padding: 0.85rem 1rem;
# }
# .qa-answer .text {
#     font-size: 0.92rem;
#     color: #2a3d1e;
#     line-height: 1.75;
#     white-space: pre-wrap;
# }
# .qa-sources {
#     border-top: 1px solid #eaf5d8;
#     padding: 0.55rem 1rem;
#     background: #fafff6;
#     display: flex;
#     flex-wrap: wrap;
#     gap: 0.4rem;
#     align-items: center;
# }
# .source-label {
#     font-size: 0.72rem;
#     color: #888;
#     font-weight: 600;
#     text-transform: uppercase;
#     letter-spacing: 0.4px;
#     margin-right: 0.2rem;
# }
# .source-chip {
#     font-size: 0.72rem;
#     background: #e8f5d4;
#     color: #2e6010;
#     border: 1px solid #c4e09a;
#     border-radius: 20px;
#     padding: 2px 9px;
#     font-weight: 500;
# }

# /* ── Input ── */
# .stTextArea textarea {
#     border: 1.5px solid #c4e09a !important;
#     border-radius: 8px !important;
#     font-family: 'IBM Plex Sans', sans-serif !important;
#     font-size: 0.92rem !important;
#     background: #ffffff !important;
#     color: #1e3d0c !important;
#     resize: none !important;
# }
# .stTextArea textarea:focus {
#     border-color: #4e9a1a !important;
#     box-shadow: 0 0 0 2px rgba(78,154,26,0.12) !important;
# }
# .stTextArea label { display: none !important; }

# /* ── Buttons ── */
# div[data-testid="column"]:first-child .stButton > button {
#     background: #2e6010;
#     color: #fff;
#     border: none;
#     border-radius: 8px;
#     font-family: 'IBM Plex Sans', sans-serif;
#     font-size: 0.92rem;
#     font-weight: 600;
#     width: 100%;
#     padding: 0.55rem;
#     transition: background 0.15s;
# }
# div[data-testid="column"]:first-child .stButton > button:hover {
#     background: #1e3d0c;
# }
# div[data-testid="column"]:last-child .stButton > button {
#     background: transparent;
#     color: #888;
#     border: 1px solid #d0d0d0;
#     border-radius: 8px;
#     font-family: 'IBM Plex Sans', sans-serif;
#     font-size: 0.92rem;
#     width: 100%;
#     padding: 0.55rem;
# }
# div[data-testid="column"]:last-child .stButton > button:hover {
#     border-color: #aaa;
#     color: #444;
# }

# /* ── Empty state ── */
# .empty-state {
#     text-align: center;
#     padding: 2.5rem 1rem;
#     color: #aac490;
#     font-size: 0.9rem;
# }
# .empty-state .icon { font-size: 2.2rem; margin-bottom: 0.5rem; }

# /* Hide Streamlit chrome */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding-top: 0 !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── Session state ─────────────────────────────────────────────────────────────
# if "rag_chain" not in st.session_state:
#     with st.spinner("Loading model and connecting to database..."):
#         st.session_state.rag_chain = build_rag_chain()

# if "history" not in st.session_state:
#     st.session_state.history = []

# # ── Header ────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="site-header">
#     <div class="site-title">🌾 <span>Agri-RAG</span> Chatbot</div>
#     <p class="site-sub">
#         AI-powered answers on <b>post-harvest crop management</b> —
#         storage, preservation &amp; loss reduction, sourced directly from agricultural documents.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # ── Chat history ──────────────────────────────────────────────────────────────
# if not st.session_state.history:
#     st.markdown("""
#     <div class="empty-state">
#         <div class="icon">🌱</div>
#         Ask a question about post-harvest management to get started.
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     for item in st.session_state.history:
#         # Build source — unique filenames only
#         source_html = ""
#         if item.get("sources"):
#             seen = set()
#             names = []
#             for s in item["sources"]:
#                 fname = s.get("file_name", "Unknown")
#                 if fname not in seen:
#                     seen.add(fname)
#                     names.append(fname)
#             source_text = ", ".join(names)
#             source_html = f"""
#             <div class="qa-sources">
#                 <span class="source-label">Source:</span>
#                 <span class="source-chip">📄 {source_text}</span>
#             </div>"""

#         st.markdown(f"""
#         <div class="qa-block">
#             <div class="qa-question">
#                 <span class="icon">Q</span>
#                 <span class="text">{item['question']}</span>
#             </div>
#             <div class="qa-answer">
#                 <div class="text">{item['answer']}</div>
#             </div>
#             {source_html}
#         </div>
#         """, unsafe_allow_html=True)

# # ── Input ─────────────────────────────────────────────────────────────────────
# st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# user_question = st.text_area(
#     "question",
#     placeholder="e.g. what is the process of making potato puree?",
#     height=90,
# )

# col1, col2 = st.columns([4, 1])
# with col1:
#     ask_clicked = st.button("Ask")
# with col2:
#     clear_clicked = st.button("Clear")

# # ── Actions ───────────────────────────────────────────────────────────────────
# if clear_clicked:
#     st.session_state.history = []
#     st.rerun()

# if ask_clicked:
#     if not user_question.strip():
#         st.warning("Please type a question.")
#     else:
#         with st.spinner("Searching documents..."):
#             result = st.session_state.rag_chain(user_question)
#         st.session_state.history.append({
#             "question": user_question,
#             "answer":   result["answer"],
#             "sources":  result["sources"],
#         })
#         st.rerun()

"""
app.py - Agri-RAG Chatbot (Streamlit Frontend)
Talks to FastAPI backend at http://localhost:8000
Run with: streamlit run app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri-RAG Chatbot",
    page_icon="🌾",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #f9fdf5;
}
.site-header {
    text-align: center;
    padding: 2rem 1rem 1.4rem;
    border-bottom: 1.5px solid #d4e8bc;
    margin-bottom: 1.8rem;
}
.site-title {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1e3d0c;
    margin: 0 0 0.35rem;
}
.site-title span { color: #4e9a1a; }
.site-sub {
    font-size: 0.88rem;
    color: #607850;
    margin: 0;
    line-height: 1.5;
}
.site-sub b { color: #2e6010; }
.qa-block {
    background: #ffffff;
    border: 1px solid #daecc4;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.qa-question {
    background: #f0f9e4;
    border-bottom: 1px solid #daecc4;
    padding: 0.75rem 1rem;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}
.qa-question .icon {
    font-size: 0.8rem;
    font-weight: 700;
    color: #4e9a1a;
    background: #d6f0b2;
    border-radius: 4px;
    padding: 2px 6px;
    margin-top: 1px;
    white-space: nowrap;
    flex-shrink: 0;
}
.qa-question .text {
    font-size: 0.92rem;
    color: #1e3d0c;
    font-weight: 500;
    line-height: 1.5;
}
.qa-answer {
    padding: 0.85rem 1rem;
}
.qa-answer .text {
    font-size: 0.92rem;
    color: #2a3d1e;
    line-height: 1.75;
    white-space: pre-wrap;
}
.qa-sources {
    border-top: 1px solid #eaf5d8;
    padding: 0.55rem 1rem;
    background: #fafff6;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}
.source-label {
    font-size: 0.72rem;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.source-chip {
    font-size: 0.72rem;
    background: #e8f5d4;
    color: #2e6010;
    border: 1px solid #c4e09a;
    border-radius: 20px;
    padding: 2px 9px;
    font-weight: 500;
}
.stTextArea textarea {
    border: 1.5px solid #c4e09a !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important;
    background: #ffffff !important;
    color: #1e3d0c !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #4e9a1a !important;
    box-shadow: 0 0 0 2px rgba(78,154,26,0.12) !important;
}
.stTextArea label { display: none !important; }
div[data-testid="column"]:first-child .stButton > button {
    background: #2e6010;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    width: 100%;
    padding: 0.55rem;
}
div[data-testid="column"]:first-child .stButton > button:hover { background: #1e3d0c; }
div[data-testid="column"]:last-child .stButton > button {
    background: transparent;
    color: #888;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    width: 100%;
    padding: 0.55rem;
}
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #aac490;
    font-size: 0.9rem;
}
.empty-state .icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="site-title">🌾 <span>Agri-RAG</span> Chatbot</div>
    <p class="site-sub">
        AI-powered answers on <b>post-harvest crop management</b> —
        storage, preservation &amp; loss reduction, sourced directly from agricultural documents.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Upload + Ingest ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Manage Documents")

    # Current PDFs
    try:
        res = requests.get(f"{API_URL}/documents", timeout=5)
        if res.status_code == 200:
            docs = res.json()
            st.markdown(f"**{docs['total']} PDF(s) in knowledge base:**")
            for f in docs["files"]:
                col_name, col_del = st.columns([3, 1])
                col_name.markdown(f"📄 {f}")
                if col_del.button("✕", key=f"del_{f}"):
                    del_res = requests.delete(f"{API_URL}/documents/{f}")
                    if del_res.status_code == 200:
                        st.success(f"Deleted {f}. Re-ingest to update.")
                        st.rerun()
        else:
            st.warning("Could not fetch document list.")
    except Exception:
        st.error("⚠️ FastAPI not running. Start it with: uvicorn main:app --reload")

    st.markdown("---")

    # Upload new PDFs
    st.markdown("### ⬆️ Upload New PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Upload"):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_files
            ]
            res = requests.post(f"{API_URL}/upload", files=files_payload)
            if res.status_code == 200:
                st.success(res.json()["message"])
                st.rerun()
            else:
                st.error(res.json().get("detail", "Upload failed."))

    st.markdown("---")

    # Ingest settings
    st.markdown("### ⚙️ Ingest Settings")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=100, max_value=2000,
        value=500, step=50,
        help="Smaller = precise retrieval. Larger = more context per chunk.",
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0, max_value=500,
        value=80, step=10,
        help="Overlap between chunks. Recommended: 10–20% of chunk size.",
    )
    loader_type = st.selectbox(
        "PDF Loader",
        options=["pypdf", "pymupdf", "pdfminer"],
        help="pypdf = fast. pymupdf = tables/formatting. pdfminer = complex layouts.",
    )

    if st.button("🔄 Ingest / Re-ingest"):
        with st.spinner("Embedding and uploading to Qdrant..."):
            res = requests.post(f"{API_URL}/ingest", json={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "loader_type": loader_type,
            })
        if res.status_code == 200:
            data = res.json()
            st.success(f"✅ {data['message']}")
            st.markdown(f"""
            - **Chunks created:** {data['total_chunks']}
            - **Files ingested:** {', '.join(data['files_ingested'])}
            - **Avg chunk size:** {data['avg_chunk_size']} chars
            """)
        else:
            st.error(res.json().get("detail", "Ingestion failed."))

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🌱</div>
        Upload PDFs → Ingest → Ask questions to get started.
    </div>
    """, unsafe_allow_html=True)
else:
    for item in st.session_state.history:
        source_html = ""
        if item.get("sources"):
            seen = set()
            names = []
            for s in item["sources"]:
                if s not in seen:
                    seen.add(s)
                    names.append(s)
            source_text = ", ".join(names)
            source_html = f"""
            <div class="qa-sources">
                <span class="source-label">Source:&nbsp;</span>
                <span class="source-chip">📄 {source_text}</span>
            </div>"""

        st.markdown(f"""
        <div class="qa-block">
            <div class="qa-question">
                <span class="icon">Q</span>
                <span class="text">{item['question']}</span>
            </div>
            <div class="qa-answer">
                <div class="text">{item['answer']}</div>
            </div>
            {source_html}
        </div>
        """, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

user_question = st.text_area(
    "question",
    placeholder="e.g. What is the recommended storage temperature for tomatoes after harvest?",
    height=90,
)

col1, col2 = st.columns([4, 1])
with col1:
    ask_clicked = st.button("Ask")
with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if ask_clicked:
    if not user_question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching documents..."):
            try:
                res = requests.post(
                    f"{API_URL}/query",
                    json={"question": user_question},
                    timeout=60,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.history.append({
                        "question": data["question"],
                        "answer":   data["answer"],
                        "sources":  data["sources"],
                    })
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Query failed."))
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Run: uvicorn main:app --reload")