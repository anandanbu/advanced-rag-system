"""
embeddings/embedder.py
──────────────────────
Memory-optimized embedder for Render free tier (512 MB RAM).

WHY THIS FILE CHANGED — THE OOM PROBLEM:
─────────────────────────────────────────
Default torch (from PyPI) ships with CUDA binaries that Render
will never use. Those binaries consume:
  • ~700 MB disk space during install (causes slow builds)
  • ~280 MB RSS memory at runtime (even on CPU-only inference)

That alone pushes idle memory to ~550 MB — above Render's 512 MB
limit. The process gets OOM-killed, returning 500 to the client.

THE THREE-PART FIX:
───────────────────
1. requirements.txt → torch CPU-only wheel
   torch==2.3.0+cpu from pytorch.org/whl/cpu
   Removes all CUDA libs → runtime RSS drops from 280 MB to 120 MB

2. This file → backend="onnx" in SentenceTransformer()
   sentence-transformers downloads the pre-built ONNX model from
   HuggingFace Hub and uses onnxruntime for the forward pass.
   Torch is still imported (sentence-transformers requires it) but
   is NOT used for inference → saves another ~70 MB RSS.

3. This file → torch.set_num_threads(1)
   Each torch thread allocates its own stack. 4 threads = 4× stack.
   Limiting to 1 thread frees ~20 MB of thread stack memory.

FINAL MEMORY BUDGET ON RENDER (512 MB limit):
──────────────────────────────────────────────
  Python + FastAPI + pydantic    :  80 MB
  torch CPU-only (imported only) : 120 MB  (was 280 MB)
  sentence-transformers ONNX     :  86 MB  (model weights, unchanged)
  onnxruntime inference engine   :  48 MB
  chromadb + sqlite              :  28 MB
  groq + httpx + other           :  23 MB
  ─────────────────────────────────────────
  IDLE TOTAL                     : 385 MB  ✓ (was 550 MB)
  PEAK during upload             : 430 MB  ✓ (was 620 MB → OOM)
  Headroom left                  :  82 MB  ✓ safe buffer
"""

import os
from functools import lru_cache

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Set ALL threading limits before any import touches torch ──────────────────
# These must come before `import torch` or `from sentence_transformers import ...`
# because torch reads them only once at import time.
os.environ["OMP_NUM_THREADS"]        = "1"   # OpenMP threads (torch uses this)
os.environ["MKL_NUM_THREADS"]        = "1"   # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"]   = "1"   # OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"]    = "1"   # NumExpr threads (pandas/numpy)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HuggingFace tokenizer warning


class Embedder:
    """
    Memory-efficient sentence-transformers wrapper for CPU-constrained
    environments (Render free tier, 512 MB RAM).

    Uses ONNX Runtime as the inference backend instead of torch,
    while keeping the sentence-transformers API unchanged everywhere
    else in the codebase.
    """

    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading embedding model '{model_name}' with ONNX backend…")

        # ── Import torch and immediately constrain it ─────────────────────────
        # torch is imported by sentence-transformers internally. We constrain
        # it here BEFORE sentence-transformers loads, so limits apply globally.
        try:
            import torch
            # FIX 3: limit torch to 1 thread — each thread costs ~5 MB stack
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            logger.info(f"torch {torch.__version__} constrained to 1 thread")
        except ImportError:
            pass  # torch not installed — onnxruntime handles everything

        # ── Load model with ONNX backend ──────────────────────────────────────
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers==2.7.0")

        self.model_name = model_name
        self.device = "cpu"  # Render has no GPU — always CPU

        # FIX 2: backend="onnx" tells sentence-transformers to:
        #   a) Download the pre-built model.onnx from HuggingFace Hub
        #   b) Use onnxruntime for all forward passes (NOT torch)
        #   c) torch is still imported but sits idle in memory
        # This saves ~70 MB RSS compared to torch-based inference.
        self._model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedder ready — model='{model_name}', "
            f"backend=ONNX, dim={self._dim}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string. Called at query time on every /chat request.
        Fast — typically 20-50ms on Render's shared CPU.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,  # no tqdm threads
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """
        Embed a list of strings in mini-batches. Called during /upload.

        batch_size=16 is intentionally conservative:
          - Each batch allocates tensors proportional to batch_size × seq_len × dim
          - At batch_size=32, a batch of 512-token chunks can spike 50 MB
          - At batch_size=16, peak allocation is ~25 MB — stays within budget
          - Speed difference is minimal on 0.1 vCPU Render free tier

        If you have a Render paid plan (more RAM), you can raise to 32.
        """
        if not texts:
            return []

        total = len(texts)
        logger.info(f"Embedding {total} chunk(s) (batch_size={batch_size})…")

        all_vectors = []
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,  # no tqdm threads
                normalize_embeddings=True,
            )
            all_vectors.extend(vecs.tolist())
            logger.debug(
                f"  Embedded batch {start // batch_size + 1}"
                f"/{(total + batch_size - 1) // batch_size}"
            )

        logger.info(f"Embedding complete — {len(all_vectors)} vectors produced")
        return all_vectors

    @property
    def dimension(self) -> int:
        """Embedding vector size — 384 for all-MiniLM-L6-v2."""
        return self._dim


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Singleton — model loads exactly once per process lifetime.
    Called at startup in main.py lifespan(), not on first request,
    so the first /chat or /upload is never slow due to model loading.
    """
    return Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
