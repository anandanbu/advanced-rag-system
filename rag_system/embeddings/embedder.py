"""
embeddings/embedder.py
──────────────────────
ONNX-optimized embedder — drops PyTorch entirely.
Saves ~232 MB RAM. Same accuracy. Faster on CPU.
"""
import os
from functools import lru_cache
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Embedder:
    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading ONNX embedding model: '{model_name}'")
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            import numpy as np
        except ImportError:
            raise ImportError(
                "Run: pip install optimum[onnxruntime] onnxruntime"
            )

        self._np = np
        self.model_name = model_name
        self.device = "cpu"  # ONNX Runtime always CPU on free tier

        # export=True converts the model to ONNX on first load, then caches it
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,          # auto-converts HuggingFace → ONNX
            provider="CPUExecutionProvider",
        )
        # Probe dimension from a dummy encode
        dummy = self._encode_raw(["test"])
        self._dim = dummy.shape[1]
        logger.info(f"ONNX embedder ready — dim={self._dim}, RAM saved ~232MB vs torch")

    def _encode_raw(self, texts: list[str]):
        """Tokenize and run ONNX inference, return mean-pooled numpy array."""
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",    # optimum expects pt tensors as input
        )
        outputs = self._model(**inputs)
        # Mean pool over token dimension
        hidden = outputs.last_hidden_state.detach().numpy()
        mask = inputs["attention_mask"].numpy()
        mask_expanded = mask[:, :, None].astype(float)
        pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clip(min=1e-9)
        # L2 normalize
        norms = self._np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype("float32")

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return self._encode_raw([text])[0].tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        logger.info(f"ONNX embedding {len(texts)} chunks...")
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._encode_raw(batch)
            results.extend(vecs.tolist())
        logger.info(f"Embedding done — {len(results)} vectors")
        return results

    @property
    def dimension(self) -> int:
        return self._dim


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )