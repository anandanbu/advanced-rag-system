"""
streamlit_app.py
─────────────────
Streamlit frontend — optimized for Render free tier backend.

Changes from original:
  - API_BASE reads from st.secrets or env var (not hardcoded localhost)
  - Wake-up ping on load — Render free tier sleeps after 15 min idle
  - Clearer empty-store warning with re-upload guidance
  - Memory-safe: no large objects stored in session_state
  - critic_score display fixed: 0.0 now shows as "0.00" not "—"
  - Timeout raised to 90s for slow cold-start responses
"""

import os
import time
import requests
import streamlit as st

# ── API Base URL ──────────────────────────────────────────────────────────────
# Priority: st.secrets → environment variable → localhost fallback
try:
    API_BASE = st.secrets["API_BASE_URL"].rstrip("/")
except Exception:
    API_BASE = os.environ.get("API_BASE_URL", "https://rag-backend-50u3.onrender.com").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False


# ── Helper functions ──────────────────────────────────────────────────────────

def wake_backend() -> bool:
    """
    Ping /health to wake Render from sleep.
    Render free tier sleeps after 15 min idle — first request takes 30-60s.
    Returns True if backend is reachable.
    """
    for attempt in range(3):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=15)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(3)
    return False


def api_chat(message: str, use_critic: bool = True) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "use_critic": use_critic,
            },
            timeout=90,   # raised from 60 — Groq + critic can take 60-80s on free tier
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {API_BASE}. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out (90s). The backend may be overloaded. Try again."}
    except Exception as e:
        return {"error": str(e)}


def api_upload(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes)},
            timeout=180,  # uploads embed all chunks — can take 2-3 min on free tier
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. Try a smaller file (< 5 pages) on free tier."}
    except Exception as e:
        return {"error": str(e)}


def api_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/upload/stats", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def format_score(score: float, mode: str) -> str:
    """
    Fixed: original showed '—' for score=0.0 because `if score` is False.
    Now shows actual score or 'N/A' only when mode is conversational.
    """
    if mode == "conversational":
        return "N/A"
    return f"{score:.2f}"


# ── Backend wake-up on first load ─────────────────────────────────────────────
if not st.session_state.backend_ready:
    with st.spinner("🔄 Connecting to backend (may take 30-60s if just waking up)…"):
        ready = wake_backend()
        st.session_state.backend_ready = ready


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption(f"Backend: `{API_BASE}`")

    # ── Health ────────────────────────────────────────────────────────────────
    health = api_health()
    if health.get("status") == "ok":
        st.success("✅ Backend connected")
        mem = health.get("memory", {})
        if mem:
            used = mem.get("used_mb", 0)
            limit = mem.get("limit_mb", 512)
            headroom = mem.get("headroom_mb", 0)
            st.progress(used / limit, text=f"RAM: {used:.0f} / {limit} MB ({headroom:.0f} MB free)")
    else:
        st.error("❌ Backend not reachable")
        st.info("Render free tier sleeps after 15 min idle.\nClick the button below to wake it.")
        if st.button("🔄 Wake backend", use_container_width=True):
            with st.spinner("Waking up backend…"):
                if wake_backend():
                    st.session_state.backend_ready = True
                    st.rerun()
                else:
                    st.error("Still unreachable. Check your Render dashboard.")

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("📄 Upload Document")
    st.caption("PDF, TXT, DOCX, CSV · Max 10 MB recommended on free tier")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        st.caption(f"File size: {file_size_mb:.1f} MB")

        if file_size_mb > 15:
            st.warning("⚠️ Large file — may timeout on Render free tier. Try < 5 MB.")

        if st.button("🚀 Upload & Ingest", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'… (this can take 1-3 min)"):
                result = api_upload(uploaded_file.getvalue(), uploaded_file.name)

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                s = result.get("stats", {})
                st.success("✅ Document ingested!")
                st.json({
                    "pages": s.get("pages_loaded"),
                    "chunks_added": s.get("chunks_added"),
                    "chunks_total": s.get("chunks_created"),
                })
                # Refresh stats
                st.session_state.vs_stats = api_stats()

    st.divider()

    # ── Knowledge Base Stats ──────────────────────────────────────────────────
    st.subheader("📊 Knowledge Base")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.vs_stats = api_stats()

    if "vs_stats" not in st.session_state:
        st.session_state.vs_stats = api_stats()

    vstats = st.session_state.vs_stats
    if vstats:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", vstats.get("total_chunks", 0))
        col2.metric("Files", len(vstats.get("sources", [])))

        if vstats.get("total_chunks", 0) == 0:
            st.warning(
                "⚠️ No documents stored.\n\n"
                "Upload a file above to get started.\n\n"
                "If you already uploaded, the backend may have restarted "
                "(Render free tier clears storage on restart). "
                "Please re-upload your file."
            )
        else:
            st.caption("**Indexed files:**")
            for src in vstats.get("sources", []):
                st.caption(f"• {src}")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    use_critic = st.toggle(
        "Quality Evaluation",
        value=True,
        help="Evaluates answer quality. Disable for faster responses on free tier.",
    )
    if use_critic:
        st.caption("⚡ Critic adds 1 extra Groq API call per response.")

    st.divider()

    # ── Session ───────────────────────────────────────────────────────────────
    st.subheader("💬 Session")
    st.caption(f"ID: `{st.session_state.session_id[:16]}…`")

    col1, col2 = st.columns(2)
    if col1.button("🗑️ New", use_container_width=True):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    if col2.button("🧹 Clear", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.post(
                f"{API_BASE}/chat/clear",
                json={"session_id": st.session_state.session_id},
                timeout=10,
            )
        except Exception:
            pass
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🧠 RAG Assistant")
st.caption(
    "Upload a document in the sidebar, then ask questions about it. "
    "Try: *summarize this document*, *what are the key points?*, *explain section 2*"
)

# Warn if no documents
vstats = st.session_state.get("vs_stats", {})
if vstats and vstats.get("total_chunks", 0) == 0:
    st.info(
        "👆 No documents uploaded yet. Use the sidebar to upload a PDF, TXT, DOCX, or CSV. "
        "Then ask any question about it."
    )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            mode = meta.get("mode", "rag")
            score = meta.get("critic_score", 0.0)
            hallucinated = meta.get("hallucination_detected", False)
            iterations = meta.get("improvement_iterations", 0)
            latency = meta.get("latency_ms", 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            # FIXED: use format_score() instead of `if score` which treated 0.0 as falsy
            c2.metric("Quality", format_score(score, mode))
            c3.metric("Hallucination", "⚠️ Yes" if hallucinated else "✅ No")
            c4.metric("Latency", f"{latency:.0f}ms")

            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunk(s) used"):
                    for i, src in enumerate(sources, 1):
                        meta_data = src.get("metadata", {})
                        src_name = meta_data.get("source", "Unknown")
                        page = meta_data.get("page", "")
                        page_str = f" · page {page}" if page else ""
                        score_val = src.get("score", 0)
                        st.markdown(
                            f"**[{i}] {src_name}{page_str}** "
                            f"(relevance: {score_val:.3f})"
                        )
                        st.text(src.get("text", "")[:300] + "…")
                        if i < len(sources):
                            st.divider()


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask anything about your uploaded document… "
    "(e.g. 'summarize this', 'what are the main points?')"
):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking… (may take 15-30s on free tier)"):
            result = api_chat(prompt, use_critic=use_critic)

        if "error" in result:
            err_msg = f"❌ Error: {result['error']}"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            answer = result.get("answer", "No response received.")
            st.markdown(answer)

            mode = result.get("mode", "rag")
            score = result.get("critic_score", 0.0)
            hallucinated = result.get("hallucination_detected", False)
            iterations = result.get("improvement_iterations", 0)
            latency = result.get("latency_ms", 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            c2.metric("Quality", format_score(score, mode))
            c3.metric("Hallucination", "⚠️ Yes" if hallucinated else "✅ No")
            c4.metric("Latency", f"{latency:.0f}ms")

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunk(s) used"):
                    for i, src in enumerate(sources, 1):
                        meta_data = src.get("metadata", {})
                        src_name = meta_data.get("source", "Unknown")
                        page = meta_data.get("page", "")
                        page_str = f" · page {page}" if page else ""
                        score_val = src.get("score", 0)
                        st.markdown(
                            f"**[{i}] {src_name}{page_str}** "
                            f"(relevance: {score_val:.3f})"
                        )
                        st.text(src.get("text", "")[:300] + "…")
                        if i < len(sources):
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": result,
            })
