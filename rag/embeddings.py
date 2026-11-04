"""
embeddings.py -- the embedding layer for the RAG vector store.

Same "real implementation + tested offline floor" discipline as
llm_narrative.py and escalation_judge.py, updated after a round-table
review flagged that the original default (OfflineTfidfEmbeddings, a
TF-IDF vectorizer) does not actually satisfy a JD line asking for
real semantic embeddings: TF-IDF matches on shared vocabulary, not
meaning, so a query using a synonym for a document's words retrieves
nothing even when the document is exactly the right answer.

LocalSentenceTransformerEmbeddings below is the real fix: a small,
pretrained neural embedding model (sentence-transformers/
all-MiniLM-L6-v2, 384 dims) running fully locally once downloaded --
no API key, no per-call network. An earlier version of this file
said huggingface.co was blocked by this sandbox's egress allowlist;
re-tested on 2026-11-04, the download succeeded (see
tests/test_embeddings.py's semantic-retrieval regression test, which
proves this isn't just "it imports" -- it retrieves a document whose
text shares no vocabulary with the query, which TF-IDF provably
cannot do). If that ever regresses in a different environment,
get_embeddings() still falls back to OfflineTfidfEmbeddings rather
than failing the whole pipeline -- but it logs that fallback instead
of taking it silently, so "why did retrieval quality drop" has an
answer in the logs.

OfflineTfidfEmbeddings is kept as that tested, no-network floor and
is still what every test that doesn't need semantic matching runs
against (fast, deterministic, no model weights to load). It is not
a stand-in that gets swapped out before anything works -- it is a
real, exercised fallback path, not dead code. get_embeddings() below
is the one seam a real embedding provider plugs into with no other
code changes, same role as llm_narrative.generate_brief's _MODEL
constant.
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class OfflineTfidfEmbeddings(Embeddings):
    """Deterministic, no-network embedding function. Fits its TF-IDF
    vocabulary from the first (and only ever the first) batch of
    documents it is asked to embed (the full corpus, built once at
    vector-store construction time -- see vector_store.py), then
    reuses that fitted vocabulary for every later query. A query
    embedded before any documents have been fitted is a programming
    error, not a data problem, so it raises rather than silently
    returning a zero vector.

    One instance is good for exactly one corpus. A round-table review
    found that a *second* call to embed_documents (e.g. from a future
    incremental vector_store.add_texts() caller reusing an already-
    fitted instance) would silently call .transform() with the old
    vocabulary instead of refitting -- any term only present in the
    new batch would vanish with no error, degrading retrieval quality
    with nothing in the logs to explain why. Since this class has no
    legitimate incremental-fit use case (the whole point of fitting
    once over the full corpus is that TF-IDF's own weighting depends
    on the full document frequency distribution), embed_documents now
    raises on a second call instead of guessing: build a fresh
    OfflineTfidfEmbeddings() per corpus build, same as
    vector_store.build_vector_store already does."""

    def __init__(self, max_features: int = 4096) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._fitted:
            raise RuntimeError(
                "OfflineTfidfEmbeddings.embed_documents called a second time on "
                "an already-fitted instance -- this would silently reuse a stale "
                "TF-IDF vocabulary and drop any new term with no error. Build a "
                "fresh OfflineTfidfEmbeddings() (or call "
                "vector_store.build_vector_store() again) for a new/changed "
                "document set instead of reusing this one."
            )
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError(
                "OfflineTfidfEmbeddings.embed_query called before embed_documents "
                "fitted a vocabulary -- build the vector store from the full "
                "corpus first (see vector_store.build_vector_store)."
            )
        return self._vectorizer.transform([text]).toarray()[0].tolist()


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Real semantic embeddings via a local, pretrained
    sentence-transformers model -- no API key, no per-call network
    (only the one-time model-weights download, cached on disk after
    the first load). This is what actually implements "implement
    embeddings, semantic search" rather than approximating it: the
    vectors encode meaning learned from the model's own pretraining,
    not just which words a document and a query happen to share.

    Stateless across calls in the way that matters for correctness --
    unlike OfflineTfidfEmbeddings, there is no fitted vocabulary to
    go stale, so embed_documents can be called more than once (e.g.
    for an incremental add_texts()) with no risk of the silent-
    staleness bug that class's docstring describes. The model itself
    is loaded once in __init__ and reused for every embed call."""

    def __init__(self, model_name: str = _LOCAL_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


def get_embeddings() -> Embeddings:
    """The one seam a real embedding provider plugs into.

    Preference order, each with a stated, logged reason for falling
    through rather than a silent downgrade:
      1. OpenAI's real embedding API, if OPENAI_API_KEY is set.
      2. The local sentence-transformers model -- real semantic
         embeddings, no API key required. This is the default in
         every environment this project has actually been run in as
         of 2026-11-04 (see embeddings.py's module docstring for the
         history of why TF-IDF used to be the default instead).
      3. OfflineTfidfEmbeddings, only if loading the local model
         itself fails (no network for the one-time download, the
         optional `sentence-transformers` package missing, or
         anything else) -- logged, not silent, so a retrieval-quality
         regression has a traceable cause.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            return OpenAIEmbeddings()
        except ImportError:
            logger.warning(
                "OPENAI_API_KEY is set but langchain-openai is not installed; "
                "falling through to the local sentence-transformers embedder."
            )

    try:
        return LocalSentenceTransformerEmbeddings()
    except Exception:  # noqa: BLE001 -- degrade, never fail vector-store build
        logger.exception(
            "LocalSentenceTransformerEmbeddings failed to load (no network for "
            "the model download, missing dependency, or similar); falling back "
            "to the deterministic, no-network OfflineTfidfEmbeddings. Retrieval "
            "will be keyword-based, not semantic, until this is fixed."
        )
        return OfflineTfidfEmbeddings()
