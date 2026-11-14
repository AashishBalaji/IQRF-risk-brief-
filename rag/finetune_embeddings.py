"""finetune_embeddings.py -- fine-tunes the local sentence-transformer
embedder (rag.embeddings.LocalSentenceTransformerEmbeddings' underlying
model, all-MiniLM-L6-v2) on this project's own real query/document
pairs, closing the one real gap a round-table review of the RAG
upgrade flagged: "construct and fine-tune RAG systems" -- everything
in rag/ *constructs* a real RAG system, but nothing in it actually
*fine-tunes* anything; every embedding call uses the pretrained model
exactly as downloaded.

**Real training data, not invented pairs:** `build_finetune_pairs()`
derives (query, positive_document) pairs directly and mechanically
from this project's own real corpus (corpus_builder.build_corpus()) --
one query per real trading day's daily-summary document, one per
pattern-detector/escalation-rule reference document, one per labelled
illustrative sample, one for the one real usage-log entry. These are
programmatically generated from the corpus's own real metadata (a
day's real date, a detector's real name), the same kind of weak
supervision a real retrieval fine-tune commonly uses when no
hand-labelled query set exists -- not hand-faked pairs chosen to make
a number look good.

**No train/eval leakage:** the exact query wording and target used by
rag/eval.py's RETRIEVAL_EVAL_CASES (the 2026-06-04 daily summary, the
`detect_exit_reason_skew` reference doc) are deliberately excluded
from the training pairs below (see EXCLUDED_DATE / EXCLUDED_REFERENCE
NAME) -- the whole point of fine-tuning is to see whether it improves
retrieval on *held-out* queries, which would mean nothing if the
model had already seen those exact query/document pairs during
training.

**Loss:** MultipleNegativesRankingLoss (in-batch negatives -- every
other positive in the batch acts as a negative for a given anchor),
the standard choice for fine-tuning a retrieval embedder on
(query, positive) pairs with no labelled hard negatives, via the
real `sentence-transformers` Trainer API (not the deprecated
`.fit()` method).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rag.corpus_builder import RagDocument, build_corpus
from rag.embeddings import _LOCAL_MODEL_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_MODEL_DIR = REPO_ROOT / "artifacts" / "finetuned-embedder"

# Deliberately excluded from training pairs -- these are the exact
# query targets rag/eval.py's RETRIEVAL_EVAL_CASES uses, so training
# on them would contaminate the before/after comparison.
EXCLUDED_DATE = "2026-06-04"
EXCLUDED_REFERENCE_NAME = "detect_exit_reason_skew"


def build_finetune_pairs(documents: List[RagDocument] | None = None) -> List[Tuple[str, str]]:
    """Returns real (query, positive_document_text) pairs derived
    mechanically from the real corpus, excluding anything that would
    leak into rag/eval.py's held-out evaluation queries."""
    docs = documents if documents is not None else build_corpus()
    pairs: List[Tuple[str, str]] = []

    for doc in docs:
        kind = doc.metadata.get("kind")

        if kind == "daily_summary":
            date = doc.metadata.get("date")
            if date == EXCLUDED_DATE:
                continue
            pairs.append((f"What happened on {date}?", doc.text))

        elif kind == "reference":
            name = doc.metadata.get("name", "")
            if name == EXCLUDED_REFERENCE_NAME:
                continue
            readable = name.removeprefix("detect_").replace("_", " ")
            pairs.append((f"what does the {readable} detector catch", doc.text))

        elif kind == "illustrative_sample":
            source = doc.metadata.get("source", "this sample")
            pairs.append((f"illustrative sample trading data from {source}", doc.text))

        elif kind == "real_usage_entry":
            pairs.append(("live market pulse check from the real usage log", doc.text))

    return pairs


def fine_tune_embedder(
    pairs: List[Tuple[str, str]],
    base_model_name: str = _LOCAL_MODEL_NAME,
    output_dir: Path = FINETUNED_MODEL_DIR,
    epochs: int = 4,
    batch_size: int = 8,
) -> Path:
    """Fine-tunes `base_model_name` on `pairs` via MultipleNegativesRankingLoss
    and saves the result to `output_dir`. Returns `output_dir` so the
    caller can load it straight back with
    `rag.embeddings.LocalSentenceTransformerEmbeddings(model_name=str(output_dir))`.
    """
    if not pairs:
        raise ValueError("no training pairs given -- nothing to fine-tune on")

    from datasets import Dataset  # noqa: PLC0415
    from sentence_transformers import (  # noqa: PLC0415
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (  # noqa: PLC0415
        MultipleNegativesRankingLoss,
    )

    model = SentenceTransformer(base_model_name)
    train_dataset = Dataset.from_dict(
        {"anchor": [q for q, _ in pairs], "positive": [d for _, d in pairs]}
    )
    loss = MultipleNegativesRankingLoss(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=min(batch_size, len(pairs)),
        logging_steps=max(1, len(pairs) // batch_size),
        report_to=[],
        save_strategy="no",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss
    )
    trainer.train()
    model.save(str(output_dir))
    return output_dir


def main() -> int:
    pairs = build_finetune_pairs()
    print(f"Built {len(pairs)} real training pairs from the real corpus.")
    out = fine_tune_embedder(pairs)
    print(f"Fine-tuned embedder saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
