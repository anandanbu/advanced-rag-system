"""
streamlit_app.py
─────────────────
Streamlit frontend for the RAG system.

Features:
  - Upload documents (PDF/TXT/DOCX/CSV)
  - Chat interface with conversation history
  - Source citations display
  - Critic score indicator
  - Session management
  - Vector store stats

Run:
  streamlit run streamlit_app.py

Make sure the FastAPI backend is running first:
  uvicorn main:app --reload
"""

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helper Functions ──────────────────────────────────────────────────────────

def api_chat(message: str, use_critic: bool = True) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "use_critic": use_critic,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is `uvicorn main:app` running?"}
    except Exception as e:
        return {"error": str(e)}


def api_upload(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes)},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/upload/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable"}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 RAG System")
    st.caption("Advanced RAG with Memory & Self-Improvement")

    # Health status
    health = api_health()
    if health.get("status") == "ok":
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend unreachable")
        st.info("Start the backend:\n```\nuvicorn main:app --reload\n```")

    st.divider()

    # ── Upload Section ────────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, DOCX, or CSV",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'…"):
                result = api_upload(uploaded_file.read(), uploaded_file.name)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                stats = result.get("stats", {})
                st.success(f"✅ Ingested successfully!")
                st.json({
                    "chunks_added": stats.get("chunks_added"),
                    "pages_loaded": stats.get("pages_loaded"),
                    "chunks_created": stats.get("chunks_created"),
                })

    st.divider()

    # ── Vector Store Stats ────────────────────────────────────────────────────
    st.subheader("📊 Knowledge Base")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.session_state.vs_stats = api_stats()

    if "vs_stats" not in st.session_state:
        st.session_state.vs_stats = api_stats()

    vstats = st.session_state.vs_stats
    if vstats:
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks", vstats.get("total_chunks", 0))
        col2.metric("Sources", len(vstats.get("sources", [])))
        if vstats.get("sources"):
            st.caption("**Indexed sources:**")
            for src in vstats["sources"]:
                st.caption(f"• {src}")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    use_critic = st.toggle("Enable Critic Evaluation", value=True)
    st.caption("Critic scores answers and triggers self-improvement if quality is low.")

    st.divider()

    # ── Session Controls ──────────────────────────────────────────────────────
    st.subheader("💬 Session")
    st.code(f"ID: {st.session_state.session_id[:16]}…", language=None)

    if st.button("🗑️ New Session", use_container_width=True):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.post(f"{API_BASE}/chat/clear", json={"session_id": st.session_state.session_id})
        except Exception:
            pass
        st.rerun()


# ── Main Chat Interface ────────────────────────────────────────────────────────

st.title("🧠 Advanced RAG Assistant")
st.caption("Upload documents → Ask questions → Get grounded, cited answers")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            mode = meta.get("mode", "rag")
            score = meta.get("critic_score", 0)
            hallucinated = meta.get("hallucination_detected", False)
            iterations = meta.get("improvement_iterations", 0)
            latency = meta.get("latency_ms", 0)

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            c2.metric("Quality Score", f"{score:.2f}" if score else "—")
            c3.metric("Hallucination", "⚠️ Yes" if hallucinated else "✅ No")
            c4.metric("Iterations", iterations)

            # Sources
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} source chunk(s)"):
                    for i, src in enumerate(sources, 1):
                        src_name = src.get("metadata", {}).get("source", "Unknown")
                        page = src.get("metadata", {}).get("page", "")
                        page_str = f" · Page {page}" if page else ""
                        st.markdown(f"**[{i}] {src_name}{page_str}** (score: {src.get('score', 0):.3f})")
                        st.text(src.get("text", "")[:400] + "…")
                        if i < len(sources):
                            st.divider()


# ── Chat Input ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = api_chat(prompt, use_critic=use_critic)

        if "error" in result:
            st.error(f"Error: {result['error']}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
        else:
            answer = result.get("answer", "No response received.")
            st.markdown(answer)

            # Metrics
            mode = result.get("mode", "rag")
            score = result.get("critic_score", 0)
            hallucinated = result.get("hallucination_detected", False)
            iterations = result.get("improvement_iterations", 0)
            latency = result.get("latency_ms", 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            c2.metric("Quality Score", f"{score:.2f}" if score else "—")
            c3.metric("Hallucination", "⚠️ Yes" if hallucinated else "✅ No")
            c4.metric("Iterations", iterations)

            # Sources
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} source chunk(s)"):
                    for i, src in enumerate(sources, 1):
                        src_name = src.get("metadata", {}).get("source", "Unknown")
                        page = src.get("metadata", {}).get("page", "")
                        page_str = f" · Page {page}" if page else ""
                        st.markdown(f"**[{i}] {src_name}{page_str}** (score: {src.get('score', 0):.3f})")
                        st.text(src.get("text", "")[:400] + "…")
                        if i < len(sources):
                            st.divider()

            # Save to session with metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": result,
            })
