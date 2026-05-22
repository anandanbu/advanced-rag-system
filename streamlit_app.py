"""
streamlit_app.py
─────────────────
Streamlit frontend — optimized for Render free tier backend.
"""

import os
import time
import uuid
import requests
import streamlit as st

# ── PAGE CONFIG MUST BE FIRST ────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Base URL ─────────────────────────────────────────────────────────────
# Render environment variable first
API_BASE = os.environ.get(
    "API_BASE_URL",
    "https://rag-backend-50u3.onrender.com"
).rstrip("/")

# Optional Streamlit secrets override
try:
    if "API_BASE_URL" in st.secrets:
        API_BASE = st.secrets["API_BASE_URL"].rstrip("/")
except Exception:
    pass

# ── Session state init ──────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False


# ── Helper Functions ────────────────────────────────────────────────────────

def wake_backend() -> bool:
    """
    Wake Render backend if sleeping.
    """
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
        return {
            "error": f"Cannot connect to backend at {API_BASE}"
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out (90s). Backend may be overloaded."
        }

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
        return {
            "error": "Upload timed out. Try a smaller file."
        }

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
    if mode == "conversational":
        return "N/A"

    return f"{score:.2f}"


# ── Wake backend on first load ──────────────────────────────────────────────
if not st.session_state.backend_ready:
    with st.spinner(
        "🔄 Connecting to backend (may take 30-60s if waking up)…"
    ):
        ready = wake_backend()
        st.session_state.backend_ready = ready


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("🧠 RAG Assistant")
    st.caption(f"Backend: `{API_BASE}`")

    # ── Health ──────────────────────────────────────────────────────────────
    health = api_health()

    if health.get("status") == "ok":

        st.success("✅ Backend connected")

        mem = health.get("memory", {})

        if mem:
            used = mem.get("used_mb", 0)
            limit = mem.get("limit_mb", 512)
            headroom = mem.get("headroom_mb", 0)

            st.progress(
                used / limit,
                text=f"RAM: {used:.0f}/{limit} MB ({headroom:.0f} MB free)"
            )

    else:
        st.error("❌ Backend not reachable")

        if st.button("🔄 Wake Backend", use_container_width=True):

            with st.spinner("Waking backend..."):

                if wake_backend():
                    st.session_state.backend_ready = True
                    st.rerun()

                else:
                    st.error("Backend still unreachable")

    st.divider()

    # ── Upload ──────────────────────────────────────────────────────────────
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose File",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:

        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024

        st.caption(f"File size: {file_size_mb:.1f} MB")

        if file_size_mb > 15:
            st.warning("⚠️ Large file may timeout on free tier")

        if st.button("🚀 Upload & Ingest", use_container_width=True):

            with st.spinner("Processing document..."):

                result = api_upload(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )

            if "error" in result:

                st.error(result["error"])

            else:

                s = result.get("stats", {})

                st.success("✅ Document ingested!")

                st.json({
                    "pages": s.get("pages_loaded"),
                    "chunks_added": s.get("chunks_added"),
                    "chunks_total": s.get("chunks_created"),
                })

                st.session_state.vs_stats = api_stats()

    st.divider()

    # ── Knowledge Base ──────────────────────────────────────────────────────
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
                "Upload a file to begin.\n\n"
                "Render free tier clears storage on restart."
            )

        else:

            st.caption("Indexed files:")

            for src in vstats.get("sources", []):
                st.caption(f"• {src}")

    st.divider()

    # ── Settings ────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    use_critic = st.toggle(
        "Quality Evaluation",
        value=True,
        help="Disable for faster responses."
    )

    st.divider()

    # ── Session ─────────────────────────────────────────────────────────────
    st.subheader("💬 Session")

    st.caption(
        f"ID: `{st.session_state.session_id[:16]}…`"
    )

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
                json={
                    "session_id": st.session_state.session_id
                },
                timeout=10,
            )
        except Exception:
            pass

        st.rerun()


# ── Main Chat Area ──────────────────────────────────────────────────────────

st.title("🧠 RAG Assistant")

st.caption(
    "Upload a document and ask questions about it."
)

# ── Empty KB warning ────────────────────────────────────────────────────────
vstats = st.session_state.get("vs_stats", {})

if vstats and vstats.get("total_chunks", 0) == 0:

    st.info(
        "👆 Upload a PDF, TXT, DOCX, or CSV to begin."
    )

# ── Chat history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:

            meta = msg["meta"]

            mode = meta.get("mode", "rag")
            score = meta.get("critic_score", 0.0)
            hallucinated = meta.get("hallucination_detected", False)
            latency = meta.get("latency_ms", 0)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mode", mode.upper())
            c2.metric("Quality", format_score(score, mode))
            c3.metric(
                "Hallucination",
                "⚠️ Yes" if hallucinated else "✅ No"
            )
            c4.metric("Latency", f"{latency:.0f}ms")


# ── Chat Input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask anything about your uploaded document..."
):

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = api_chat(
                prompt,
                use_critic=use_critic
            )

        if "error" in result:

            err_msg = f"❌ Error: {result['error']}"

            st.error(err_msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg
            })

        else:

            answer = result.get(
                "answer",
                "No response received."
            )

            st.markdown(answer)

            mode = result.get("mode", "rag")
            score = result.get("critic_score", 0.0)
            hallucinated = result.get(
                "hallucination_detected",
                False
            )
            latency = result.get("latency_ms", 0)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mode", mode.upper())
            c2.metric("Quality", format_score(score, mode))
            c3.metric(
                "Hallucination",
                "⚠️ Yes" if hallucinated else "✅ No"
            )
            c4.metric("Latency", f"{latency:.0f}ms")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": result,
            })
