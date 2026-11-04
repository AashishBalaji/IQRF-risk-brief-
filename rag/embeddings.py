"""
embeddings.py -- the embedding layer for the RAG vector store.

Same "real implementation + tested offline floor" discipline as
llm_narrative.py and escalation_judge.py: a real embedding provider
(OpenAI's or a local HuggingFace sentence-transformer) is what a
production deployment would use, but neither can be verified live in
this sandbox -- OPENAI_API_KEY isn't available here, and
huggingface.co is blocked by this sandbox's egress allowlist (a
pretrained-model download gets a 403 at the proxy, confirmed the same
way veridex's README documents it for its own FAISS layer -- not a
guess, the same sandbox, the same block).

OfflineTfidfEmbeddings is therefore the real, tested default: a
deterministic TF-IDF vectorizer (scikit-learn) that implements
LangChain's Embeddings interface exactly, no network and no API key
required. It is not a stand-in that gets swapped out before anything
works -- the whole RAG pipeline in this project runs on it, is tested
against it, and degrades to nothing else. get_embeddings() below is
the one seam a real embedding provider would plug into later with no
other code changes, same role as llm_narrative.generate_brief's
_MODEL constant.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


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


def get_embeddings() -> Embeddings:
    """The one seam a real embedding provider plugs into. If
    OPENAI_API_KEY is set, uses OpenAI's real embedding API (requires
    the optional `langchain-openai` package); otherwise, and in every
    environment this project has actually been run in, falls back to
    the offline TF-IDF embedder -- never silently drops to a worse
    model without saying so."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            return OpenAIEmbeddings()
        except ImportError:
            pass
    return OfflineTfidfEmbeddings()
