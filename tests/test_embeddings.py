"""Smoke test against the real LaBSE model -- verified live
(GPU load ~15s, cosine(museum, musée)=0.976, cosine(museum, متحف)=0.935).
Slower than the rest of the suite by design (real model load); kept as a
single focused test rather than routing every module's tests through the
real model."""

from lad.rag.embeddings import Embedder


def test_labse_cross_lingual_similarity():
    embedder = Embedder()
    vectors = embedder.encode(["museum", "musée", "متحف", "archaeology"])

    assert vectors.shape == (4, 768)

    def cosine(a, b):
        return float(a @ b)  # already L2-normalized by Embedder.encode

    museum_musee = cosine(vectors[0], vectors[1])
    museum_arabic = cosine(vectors[0], vectors[2])
    museum_archaeology = cosine(vectors[0], vectors[3])

    # cross-lingual translations of the same word should be far more
    # similar than two different English words
    assert museum_musee > 0.9
    assert museum_arabic > 0.85
    assert museum_musee > museum_archaeology
    assert museum_arabic > museum_archaeology
