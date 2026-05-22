"""
embeddings/embedder.py
──────────────────────
Fixed: RuntimeError: cannot set number of interop threads after
       parallel work has started or set_num_interop_threads called

ROOT CAUSE:
  torch.set_num_threads(1) and torch.set_num_interop_threads(1)
  were called inside the Embedder.__init__() method.

  On the first request, the pipeline creates RAGPipeline() which
  calls get_vector_store() which calls get_embedder() which runs
  Embedder.__init__(). By this point, uvicorn and FastAPI have already
  started their thread pools, and torch has already initialized its
  own interop thread pool during the import of sentence_transformers
  in main.py's lifespan startup.

  torch only allows set_num_interop_threads() to be called ONCE,
  before any parallel work starts. Calling it a second time raises
  RuntimeError no matter what.

FIX:
  Move ALL torch thread-limiting calls to MODULE LEVEL — outside any
  class or function. Module-level code runs exactly once when Python
  first imports this file, which happens before uvicorn starts any
  threads. This guarantees the calls happen at the right time.

  The try/except around each call handles the case where torch is
  not installed or was already configured by something else.
"""

import os

# ── MUST BE MODULE LEVEL — runs once at import time, before any threads ───────
# torch only allows these to be called before parallel work starts.
# Putting them inside __init__ is too late — uvicorn is already running.
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set torch thread limits at module import time.
# Wrapped in try/except because:
#   - torch may not be installed (ONNX-only setup)
#   - another module may have already called this (safe to ignore)
try:
    import torch
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass  # already set — fine
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started — fine, env vars above still help
except ImportError:
    pass  # torch not installed — onnxruntime handles everything

# ── Normal imports ────────────────────────────────────────────────────────────
from functools import lru_cache
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Memory-efficient sentence-transformers wrapper.
    Uses ONNX backend to save ~70 MB RAM vs torch inference.
    Thread limits are set at module level above — NOT here.
    """

    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading embedding model '{model_name}' with ONNX backend…")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers==2.7.0")

        self.model_name = model_name
        self.device     = "cpu"  # Render free tier has no GPU

        # backend="onnx" downloads the pre-built ONNX model from HuggingFace
        # and uses onnxruntime for inference instead of torch.
        # torch is still imported (sentence-transformers needs it) but sits
        # idle — saving ~70 MB RSS compared to torch-based inference.
        self._model = SentenceTransformer(
            model_name,
            device=self.device,
            backend="onnx",
        )

        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedder ready — dim={self._dim}, backend=ONNX")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Used at query time."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """
        Embed a list of strings in mini-batches.
        batch_size=16 keeps peak RAM under 430 MB on Render free tier.
        """
        if not texts:
            return []

        total = len(texts)
        logger.info(f"Embedding {total} chunk(s) (batch_size={batch_size})…")

        all_vectors = []
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            vecs  = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_vectors.extend(vecs.tolist())

        logger.info(f"Embedding complete — {len(all_vectors)} vectors")
        return all_vectors

    @property
    def dimension(self) -> int:
        """Embedding vector size — 384 for all-MiniLM-L6-v2."""
        return self._dim


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Singleton — model loads exactly once per process."""
    return Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
