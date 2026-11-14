"""
vector_store.py -- builds the vector store the RAG layer retrieves
from, over the real corpus corpus_builder.py assembles.

Two real backends, named directly because a "Junior GenAI Engineer"
JD line names them directly ("vector databases such as FAISS,
Pinecone"), rather than leaving the JD's literal product names
unaddressed behind a single ChromaDB implementation:

  - build_vector_store()       -- ChromaDB (the original backend,
                                   unchanged; ships its own on-disk
                                   persistence via `persist_directory`).
  - build_faiss_vector_store() -- FAISS (`langchain_community`'s
                                   wrapper over Meta's `faiss` library),
                                   added on 2026-11-14. Pinecone itself
                                   is a hosted service that needs a paid
                                   account and an API key this sandbox
                                   doesn't have, which is exactly the
                                   same "can't independently verify"
                                   situation as the live LLM-call paths
                                   elsewhere in this project (see
                                   README's "What's actually verified"
                                   section) -- so FAISS is the one of
                                   the JD's two named options that can
                                   actually be built and tested for real
                                   here, not just described.

Both return an object satisfying LangChain's VectorStore interface
(`similarity_search`, `add_texts`, ...), so `rag/eval.py`'s retrieval
evaluation and `rag/rag_chain.py`'s retrieval step work against either
backend completely unchanged -- the backend is an implementation
choice, not something the rest of the pipeline needs to know about.

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
from langchain_community.vectorstores import FAISS

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


def build_faiss_vector_store(
    documents: Optional[List[RagDocument]] = None,
    persist_directory: Optional[str] = None,
) -> FAISS:
    """Builds the FAISS vector store from the real corpus -- same
    corpus, same embedder (`rag.embeddings.get_embeddings()`) as
    `build_vector_store()`, so the two backends are a fair, like-for-
    like comparison rather than one being handed an easier corpus or
    a different embedding model.

    FAISS has no notion of a named "collection" (unlike Chroma, which
    is a client/server store with collections as a first-class
    concept) -- a FAISS index is just an in-memory structure you can
    optionally serialize to disk, so `persist_directory`, when given,
    calls `save_local()` after building rather than being threaded
    through the constructor the way Chroma's is."""
    docs = documents if documents is not None else build_corpus()
    if not docs:
        raise ValueError("corpus is empty -- nothing to build a vector store from")

    store = FAISS.from_texts(
        texts=[d.text for d in docs],
        embedding=get_embeddings(),
        metadatas=[d.metadata for d in docs],
    )
    if persist_directory is not None:
        store.save_local(persist_directory)
    return store
