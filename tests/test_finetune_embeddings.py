"""Unit tests for rag/finetune_embeddings.py -- the real training-pair
derivation and a real (but deliberately tiny/fast) training smoke
test. The full real fine-tune (21 pairs, 4 epochs) and its real
before/after retrieval-precision comparison are documented in
README.md's "Fine-tuned embedder" section rather than re-run here on
every test invocation -- same reasoning as live_usage_check.py not
being part of the automated suite: it's a real, reproducible,
timestamped artifact-producing run, not a repeatable assertion."""

from __future__ import annotations

from rag.finetune_embeddings import (
    EXCLUDED_DATE,
    EXCLUDED_REFERENCE_NAME,
    build_finetune_pairs,
    fine_tune_embedder,
)


def test_build_finetune_pairs_derives_pairs_from_the_real_corpus():
    pairs = build_finetune_pairs()
    assert len(pairs) > 0
    for query, positive in pairs:
        assert isinstance(query, str) and query
        assert isinstance(positive, str) and positive


def test_build_finetune_pairs_excludes_the_held_out_eval_date():
    pairs = build_finetune_pairs()
    assert not any(EXCLUDED_DATE in query for query, _ in pairs)


def test_build_finetune_pairs_excludes_the_held_out_eval_reference_name():
    pairs = build_finetune_pairs()
    readable = EXCLUDED_REFERENCE_NAME.removeprefix("detect_").replace("_", " ")
    assert not any(readable in query for query, _ in pairs)


def test_build_finetune_pairs_covers_every_real_daily_summary_except_the_excluded_one():
    pairs = build_finetune_pairs()
    daily_queries = [q for q, _ in pairs if q.startswith("What happened on")]
    # 13 real trading days in the fixture, minus the one held out for eval.
    assert len(daily_queries) == 12


def test_fine_tune_embedder_raises_on_no_pairs():
    import pytest

    with pytest.raises(ValueError):
        fine_tune_embedder([])


def test_fine_tune_embedder_runs_for_real_on_a_tiny_pair_set_and_saves_a_model(tmp_path):
    """A real (not mocked) training run through the actual
    sentence-transformers Trainer -- kept tiny (4 pairs, 1 epoch) so
    it stays fast in CI; the point is proving the training mechanics
    work end to end (a real SentenceTransformer, a real
    MultipleNegativesRankingLoss step, a real save), not reproducing
    README's full 21-pair/4-epoch result here."""
    pairs = build_finetune_pairs()[:4]
    output_dir = tmp_path / "tiny-finetune"

    result_dir = fine_tune_embedder(pairs, output_dir=output_dir, epochs=1, batch_size=4)

    assert result_dir == output_dir
    assert (output_dir / "config.json").exists()
    assert (output_dir / "model.safetensors").exists()
