# 🧠 Advanced RAG System v2.0

A **production-ready Retrieval-Augmented Generation** system with memory,
critic evaluation, self-improvement loop, ReAct agent, and a full FastAPI backend.

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│   POST /chat  POST /upload  POST /agent  GET /history      │
└───────────────────┬────────────────────────────────────────┘
                    │
        ┌───────────┼────────────────┐
        │           │                │
   ┌────▼────┐  ┌───▼────┐   ┌──────▼──────┐
   │ Memory  │  │ RAG    │   │ Agent       │
   │ System  │  │Pipeline│   │ (ReAct)     │
   └─────────┘  └───┬────┘   └──────┬──────┘
                    │               │
              ┌─────▼─────┐   ┌────▼────┐
              │ ChromaDB  │   │  Tools  │
              │ Vector DB │   │ Search  │
              └─────┬─────┘   └─────────┘
                    │
              ┌─────▼─────┐
              │  Groq LLM │
              │  (Llama3) │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │  Critic   │
              │Evaluator  │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   Self-   │
              │Improvement│
              └───────────┘
```

---

## 📁 Complete Project Structure

```
rag_system/
│
├── main.py                          # FastAPI entry point
├── streamlit_app.py                 # Streamlit UI
├── requirements.txt                 # All dependencies
├── .env.example                     # Environment template
├── pytest.ini                       # Test configuration
├── Makefile                         # Developer commands
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Pydantic BaseSettings (single config)
│
├── api/
│   ├── __init__.py
│   ├── middleware.py                 # CORS, logging, error handling
│   └── routes/
│       ├── __init__.py
│       ├── chat.py                   # POST /chat
│       ├── upload.py                 # POST /upload, DELETE /upload/{src}
│       ├── history.py                # GET /history, memory endpoints
│       ├── agent.py                  # POST /agent (ReAct)
│       └── summary.py               # GET /summary/{source}
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                     # PDF / TXT / DOCX / CSV loader
│   └── chunker.py                    # Recursive text splitter + overlap
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py                   # sentence-transformers singleton
│
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py               # ChromaDB CRUD + semantic search
│
├── llm/
│   ├── __init__.py
│   └── groq_client.py                # Groq API (Llama3) + retry logic
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py                   # Core RAG orchestrator
│   ├── prompt_templates.py           # All prompts (RAG, critic, improver)
│   └── improver.py                   # Self-improvement loop
│
├── critic/
│   ├── __init__.py
│   └── evaluator.py                  # LLM-as-judge answer scoring
│
├── memory/
│   ├── __init__.py
│   ├── conversation.py               # In-session history (in-memory)
│   └── persistent.py                 # Long-term JSON disk memory
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Abstract base agent class
│   └── rag_agent.py                  # ReAct agent with tools
│
├── tools/
│   ├── __init__.py
│   ├── web_search.py                 # DuckDuckGo search (optional)
│   └── document_summary.py          # Map-reduce document summarization
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                     # Colored structured logging
│   └── helpers.py                    # Token counting, file utils, etc.
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures, env setup
│   ├── test_ingestion.py             # Loader + chunker tests
│   ├── test_embeddings.py            # Embedder tests
│   ├── test_memory.py                # Conversation + persistent memory
│   ├── test_rag_pipeline.py          # Pipeline + critic + improver tests
│   └── test_api.py                   # FastAPI endpoint tests
│
├── scripts/
│   ├── ingest_folder.py             # Bulk folder ingestion CLI
│   ├── query_cli.py                 # Interactive terminal chat
│   ├── reset_db.py                  # Reset vector store / memory
│   └── evaluate_rag.py              # Batch quality evaluation
│
├── docker/
│   ├── Dockerfile                   # Multi-stage Docker image
│   ├── docker-compose.yml           # API + UI services
│   └── .dockerignore
│
└── data/
    ├── uploads/                     # Uploaded documents
    ├── chroma_db/                   # ChromaDB persistence
    ├── memory/                      # JSON memory files
    └── sample_eval_qa.json          # Sample QA evaluation pairs
```

---

## ⚡ Quick Start (5 minutes)

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd rag_system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` — the only required value is your Groq key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

👉 Get a **free** Groq API key at: https://console.groq.com
   (Free tier: 14,400 requests/day, Llama 3 70B access)

### 3. Start the backend

```bash
uvicorn main:app --reload
# or:
make run
```

✅ API running at: http://localhost:8000
📖 Swagger docs:  http://localhost:8000/docs

### 4. Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?", "session_id": "demo"}'
```

### 6. (Optional) Start the UI

```bash
streamlit run streamlit_app.py
# or:
make run-ui
```

🎨 UI at: http://localhost:8501

---

## 🔌 Complete API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message to the RAG assistant |
| `POST` | `/chat/clear` | Reset conversation history |
| `GET`  | `/chat/session/{id}` | Get session stats |

**POST /chat body:**
```json
{
  "message": "What are the key findings?",
  "session_id": "optional-string",
  "use_critic": true,
  "filter_source": null,
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The key findings are...",
  "session_id": "abc-123",
  "sources": [{"chunk_id": "...", "text": "...", "score": 0.92, "metadata": {...}}],
  "critic_score": 0.88,
  "critic_passed": true,
  "hallucination_detected": false,
  "improvement_iterations": 0,
  "retrieval_count": 5,
  "latency_ms": 1234,
  "mode": "rag"
}
```

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/upload` | Upload and ingest a document |
| `GET`    | `/upload/stats` | Vector store statistics |
| `DELETE` | `/upload/{source_name}` | Remove a document |

### Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/agent` | Run ReAct multi-step agent |

**POST /agent body:**
```json
{
  "query": "Compare the crop yields from all uploaded reports",
  "session_id": "optional",
  "include_trace": true
}
```

### History & Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/history/sessions` | List all session IDs |
| `GET`  | `/history/{id}` | Full conversation history |
| `GET`  | `/history/{id}/memory` | Persistent memory facts |
| `POST` | `/history/{id}/memory` | Store a fact `{"key": "k", "value": "v"}` |

### Summarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/summary/{source_name}` | Summarize an ingested document |
| `POST` | `/summary/text` | Summarize arbitrary text |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + stats |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

## 🧪 Running Tests

```bash
# All tests
make test

# Fast unit tests only (no API key needed)
make test-unit

# Embedding model tests
make test-embeddings

# RAG pipeline tests (mocked)
make test-pipeline

# API endpoint tests
make test-api

# With coverage report
make test-coverage
```

---

## 🛠️ Developer Commands

```bash
make install           # Install dependencies
make run               # Start FastAPI backend
make run-ui            # Start Streamlit UI
make cli               # Interactive terminal chat
make test              # Run all tests
make lint              # Code style check
make format            # Auto-format with black
make clean             # Remove __pycache__ etc.

# Document ingestion
make ingest DIR=./my_documents
make ingest-recursive DIR=./my_docs

# Database management
make reset             # Reset vector store (with confirmation)
make eval              # Run evaluation on sample QA pairs

# Docker
make docker-up         # Build + start all services
make docker-down       # Stop services
make docker-logs       # Follow API logs
```

---

## 🐍 Python Client Example

```python
import requests

BASE = "http://localhost:8000"

# 1. Upload a document
with open("report.pdf", "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": f})
    print(r.json())
# {"message": "report.pdf ingested successfully.", "stats": {...}}

# 2. Chat with RAG
r = requests.post(f"{BASE}/chat", json={
    "message": "What are the main conclusions?",
    "session_id": "my-session",
    "use_critic": True,
})
data = r.json()
print(data["answer"])
print(f"Score: {data['critic_score']:.2f}")
print(f"Sources: {[s['metadata']['source'] for s in data['sources']]}")

# 3. Multi-step agent
r = requests.post(f"{BASE}/agent", json={
    "query": "Analyze all uploaded documents and give me a comparison",
    "include_trace": True,
})
result = r.json()
print(result["answer"])
print(f"Steps taken: {result['total_steps']}")

# 4. Save user facts
requests.post(f"{BASE}/history/my-session/memory", json={
    "key": "profession", "value": "agricultural scientist"
})

# 5. Get history
r = requests.get(f"{BASE}/history/my-session")
print(r.json()["summary"])
```

---

## 🐳 Docker Deployment

```bash
# Build and start everything
make docker-up

# Services:
# API: http://localhost:8000
# UI:  http://localhost:8501

# View logs
make docker-logs

# Stop
make docker-down
```

Data is persisted in Docker named volumes (`rag_chroma`, `rag_memory`, `rag_uploads`).

---

## 🐛 Debugging Guide

### ❌ `GROQ_API_KEY` error on startup
```
→ Run: cp .env.example .env
→ Edit .env and set GROQ_API_KEY=gsk_...
→ Get free key: https://console.groq.com
```

### ❌ Embedding model slow to load
```
→ Normal on first run — downloads ~80MB model
→ Cached locally after first download
→ Set EMBEDDING_DEVICE=cuda for GPU speedup
```

### ❌ ChromaDB sqlite3 version error
```python
# Add these 3 lines to the very top of main.py:
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
# Then: pip install pysqlite3-binary
```

### ❌ Poor retrieval / irrelevant answers
```
→ Try smaller chunks: CHUNK_SIZE=256
→ Increase retrieval: RETRIEVER_TOP_K=8
→ Check ingestion: GET /upload/stats
→ Use DEBUG logging: LOG_LEVEL=DEBUG
→ Run evaluation: python scripts/evaluate_rag.py
```

### ❌ Critic keeps triggering self-improvement
```
→ Lower threshold: CRITIC_SCORE_THRESHOLD=0.5
→ Disable critic in request: "use_critic": false
→ Check prompts in rag/prompt_templates.py
```

### ❌ 429 Rate limit from Groq
```
→ You've hit the free tier limit (30 req/min)
→ Add delays between calls
→ Use smaller model: GROQ_MODEL=llama3-8b-8192
→ Upgrade to Groq paid tier
```

### Enable verbose debug logging
```env
LOG_LEVEL=DEBUG
```

---

## 🆓 Free Tier Stack Summary

| Tool | Limit | Purpose |
|------|-------|---------|
| **Groq** | 14,400 req/day, 30/min | LLM inference |
| **sentence-transformers** | Unlimited (local) | Embeddings |
| **ChromaDB** | Unlimited (local disk) | Vector storage |
| **FastAPI** | Open source | Backend |
| **Streamlit** | Open source | Frontend |
| **DuckDuckGo** | Unlimited (web search) | Agent tool |

**Total cost for MVP: $0**

---

## 🚀 Future Upgrades Roadmap

### Phase 2 — Enhanced Retrieval
- [ ] **Hybrid search** — BM25 keyword + semantic search combined
- [ ] **Cross-encoder re-ranking** — improve top-k precision
- [ ] **Multi-vector retrieval** — HyDE (hypothetical document embeddings)
- [ ] **Metadata filtering UI** — filter by document, date, topic

### Phase 3 — Production Hardening
- [ ] **PostgreSQL** — replace JSON memory with proper database
- [ ] **Redis** — cache frequent queries and session data
- [ ] **JWT authentication** — multi-user support
- [ ] **Rate limiting** — per-user API throttling
- [ ] **Async pipeline** — fully async RAG for higher throughput
- [ ] **Streaming responses** — token-by-token streaming via SSE

### Phase 4 — AI Companion System
- [ ] **Emotional memory** — track user sentiment and mood over time
- [ ] **Proactive messaging** — scheduled check-ins via APScheduler
- [ ] **Voice interface** — Whisper STT + TTS integration
- [ ] **WhatsApp/Telegram bot** — messaging platform integration
- [ ] **Relationship graph** — track entities, people, events the user mentions

### Phase 5 — Domain Specialization (Agriculture AI)
- [ ] **Domain embeddings** — fine-tune on agricultural corpus
- [ ] **Weather tool** — real-time weather API integration
- [ ] **Crop disease vision** — image classification for disease detection
- [ ] **Market prices** — commodity price API integration
- [ ] **Soil analysis** — structured data ingestion from sensors

### Phase 6 — Autonomous Agents
- [ ] **Code execution** — sandboxed Python REPL tool
- [ ] **Web browsing** — Playwright-based browser tool
- [ ] **Email/calendar** — personal assistant integrations
- [ ] **Multi-agent orchestration** — specialist sub-agents

---

## 📚 Key Concepts Explained

**RAG (Retrieval-Augmented Generation)**
Instead of relying solely on the LLM's training data, RAG fetches
relevant passages from your documents and includes them in the prompt.
This grounds answers in your data and dramatically reduces hallucinations.

**Critic (LLM-as-Judge)**
A second LLM call evaluates the first answer for accuracy, faithfulness,
and completeness. This is inspired by Constitutional AI and RLHF research.

**Self-Improvement Loop**
If the critic score is below threshold, the system regenerates the answer
using the critic's specific feedback as guidance. This is a simplified
version of the Self-RAG paper technique.

**ReAct Agent (Reasoning + Acting)**
The agent interleaves reasoning steps with tool calls. It thinks about what
it needs, calls a tool (search, calculate, summarize), observes the result,
then reasons again — repeating until it can give a confident answer.

**Persistent Memory**
Facts about the user (location, profession, preferences) are saved to JSON
files on disk. On future sessions, these facts are injected into the system
prompt so the LLM "remembers" the user across conversations.
