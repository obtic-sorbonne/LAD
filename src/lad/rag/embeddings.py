"""Thin wrapper around sentence-transformers' LaBSE for cross-lingual
embedding. Verified live: GPU load ~15s, encodes fast,
cosine(museum, musée)=0.976, cosine(museum, متحف)=0.935 -- LaBSE captures
cross-lingual equivalence well across exactly the three languages this
project needs.

One model instance per process is expected (loading takes ~15s) -- callers
should build one Embedder and reuse it across an index-build or query run,
not construct a fresh one per call.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/LaBSE"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        """Returns L2-normalized embeddings (float32, shape [len(texts), dim])
        -- normalized so FAISS inner-product search is equivalent to cosine
        similarity (LAD paper §3.3's exact setup)."""
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype(np.float32)
