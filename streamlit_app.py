"""
streamlit_app.py
─────────────────
Fixed: removed st.secrets usage that caused the error:
  "No secrets found. Valid paths for a secrets.toml file..."

The API_BASE_URL is now read from Render's environment variables only.
No secrets.toml file needed anywhere.

HOW TO SET IT ON RENDER (Streamlit service):
  Render Dashboard → your streamlit service → Environment → Add variable:
    Key:   API_BASE_URL
    Value: https://your-backend-name.onrender.com
"""

import os
import time
import uuid
import requests
import streamlit as st

# ── API Base URL ──────────────────────────────────────────────────────────────
# Reads from Render environment variable.
# No st.secrets, no secrets.toml — just a plain env var.
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs_stats" not in st.session_state:
    st.session_state.vs_stats = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def wake_backend() -> bool:
    """Ping backend up to 3 times — Render free tier sleeps after 15 min idle."""
    for _ in range(3):
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
            timeout=90,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {API_BASE}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out (90s). Backend may be overloaded — try again."}
    except Exception as e:
        return {"error": str(e)}


def api_upload(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes)},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. Try a smaller file (< 5 MB) on free tier."}
    except Exception as e:
        return {"error": str(e)}


def api_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/upload/stats", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def format_score(score: float, mode: str) -> str:
    """Show actual number. 0.0 is NOT shown as — (that was the old bug)."""
    if mode == "conversational":
        return "N/A"
    return f"{score:.2f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption(f"Backend: `{API_BASE}`")

    # Health check
    health = api_health()
    if health.get("status") == "ok":
        st.success("✅ Backend connected")
        mem = health.get("memory", {})
        if mem:
            used = mem.get("used_mb", 0)
            limit = mem.get("limit_mb", 512)
            st.progress(
                min(used / limit, 1.0),
                text=f"RAM: {used:.0f} / {limit} MB"
            )
    else:
        st.error("❌ Backend not reachable")
        st.caption(f"Trying: `{API_BASE}`")
        if st.button("🔄 Wake backend", use_container_width=True):
            with st.spinner("Waking up backend (30–60s on free tier)…"):
                if wake_backend():
                    st.rerun()
                else:
                    st.error("Still unreachable. Check your Render dashboard.")

    st.divider()

    # Upload
    st.subheader("📄 Upload Document")
    st.caption("PDF · TXT · DOCX · CSV")

    uploaded_file = st.file_uploader(
        "Choose file",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        st.caption(f"Size: {size_mb:.1f} MB")
        if size_mb > 10:
            st.warning("Large file — may timeout on free tier. Try < 5 MB.")

        if st.button("🚀 Upload & Ingest", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'… (1–3 min)"):
                result = api_upload(uploaded_file.getvalue(), uploaded_file.name)
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                s = result.get("stats", {})
                st.success("✅ Ingested!")
                st.json({
                    "pages": s.get("pages_loaded"),
                    "chunks_added": s.get("chunks_added"),
                })
                st.session_state.vs_stats = api_stats()

    st.divider()

    # Knowledge base stats
    st.subheader("📊 Knowledge Base")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.session_state.vs_stats = api_stats()

    if not st.session_state.vs_stats:
        st.session_state.vs_stats = api_stats()

    vstats = st.session_state.vs_stats
    if vstats:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", vstats.get("total_chunks", 0))
        col2.metric("Files", len(vstats.get("sources", [])))
        if vstats.get("total_chunks", 0) == 0:
            st.warning(
                "No documents stored.\n\n"
                "Upload a file above to get started.\n\n"
                "**Note:** Render free tier clears storage on restart. "
                "Re-upload your file if the server restarted."
            )
        else:
            for src in vstats.get("sources", []):
                st.caption(f"• {src}")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    use_critic = st.toggle(
        "Quality Evaluation",
        value=True,
        help="Disable for faster responses on free tier.",
    )

    st.divider()

    # Session controls
    st.subheader("💬 Session")
    st.caption(f"ID: `{st.session_state.session_id[:16]}…`")
    col1, col2 = st.columns(2)
    if col1.button("🗑️ New", use_container_width=True):
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


# ── Main chat ─────────────────────────────────────────────────────────────────
st.title("🧠 RAG Assistant")
st.caption(
    "Upload a document in the sidebar, then ask questions. "
    "Try: *summarize this document* · *what are the key points?* · *explain section 2*"
)

# Warn if store is empty
vstats = st.session_state.vs_stats
if vstats and vstats.get("total_chunks", 0) == 0:
    st.info(
        "👆 No documents uploaded yet. "
        "Use the sidebar to upload a PDF, TXT, DOCX, or CSV file."
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            mode = meta.get("mode", "rag")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            c2.metric("Quality", format_score(meta.get("critic_score", 0.0), mode))
            c3.metric("Hallucination", "⚠️ Yes" if meta.get("hallucination_detected") else "✅ No")
            c4.metric("Latency", f"{meta.get('latency_ms', 0):.0f}ms")
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunk(s)"):
                    for i, src in enumerate(sources, 1):
                        m = src.get("metadata", {})
                        name = m.get("source", "Unknown")
                        page = m.get("page", "")
                        pg = f" · page {page}" if page else ""
                        st.markdown(f"**[{i}] {name}{pg}** (score: {src.get('score', 0):.3f})")
                        st.text(src.get("text", "")[:300] + "…")
                        if i < len(sources):
                            st.divider()

# Chat input
if prompt := st.chat_input("Ask about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = api_chat(prompt, use_critic=use_critic)

        if "error" in result:
            msg = f"❌ {result['error']}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            answer = result.get("answer", "No response.")
            st.markdown(answer)

            mode = result.get("mode", "rag")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", mode.upper())
            c2.metric("Quality", format_score(result.get("critic_score", 0.0), mode))
            c3.metric("Hallucination", "⚠️ Yes" if result.get("hallucination_detected") else "✅ No")
            c4.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunk(s)"):
                    for i, src in enumerate(sources, 1):
                        m = src.get("metadata", {})
                        name = m.get("source", "Unknown")
                        page = m.get("page", "")
                        pg = f" · page {page}" if page else ""
                        st.markdown(f"**[{i}] {name}{pg}** (score: {src.get('score', 0):.3f})")
                        st.text(src.get("text", "")[:300] + "…")
                        if i < len(sources):
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": result,
            })
