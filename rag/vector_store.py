"""
vector_store.py -- builds the ChromaDB vector store the RAG layer
retrieves from, over the real corpus corpus_builder.py assembles.

One build call embeds the whole corpus in a single batch -- this is
what fits OfflineTfidfEmbeddings' vocabulary (see embeddings.py). A
fresh collection_name is used per call by default so repeated builds
(tests, a long-running API process restarting) never collide with a
stale collection from a previous run.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from langchain_chroma import Chroma

from rag.corpus_builder import RagDocument, build_corpus
from rag.embeddings import get_embeddings


def build_vector_store(
    documents: Optional[List[RagDocument]] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """Builds (or rebuilds) the Chroma vector store from the real
    corpus. Pass `documents` explicitly in tests to avoid re-deriving
    the full corpus every time; omit it to build from the project's
    real data (corpus_builder.build_corpus())."""
    docs = documents if documents is not None else build_corpus()
    if not docs:
        raise ValueError("corpus is empty -- nothing to build a vector store from")

    return Chroma.from_texts(
        texts=[d.text for d in docs],
        embedding=get_embeddings(),
        metadatas=[d.metadata for d in docs],
        collection_name=collection_name or f"iqrf-risk-brief-{uuid.uuid4().hex[:8]}",
        persist_directory=persist_directory,
    )
