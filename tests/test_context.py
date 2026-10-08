"""Tests for Task 2.4: increased forward context in LLM refinement."""

from app.core import pipeline
from app.core.pipeline import REFINE_CONTEXT_AFTER, REFINE_CONTEXT_BEFORE, REFINE_VERSION


def test_refine_context_constants():
    assert REFINE_CONTEXT_BEFORE == 6
    assert REFINE_CONTEXT_AFTER == 8
    assert REFINE_VERSION >= 7


def test_following_context_slice_length():
    units = [{"id": i, "text": f"text {i}", "translation": f"trans {i}"} for i in range(50)]
    block = units[0:25]
    first = 0
    following = [{"source": u["text"]}
                 for u in units[first + len(block):first + len(block) + REFINE_CONTEXT_AFTER]]
    assert len(following) == 8
    assert [u["source"] for u in following] == [f"text {i}" for i in range(25, 33)]

    # Near end of episode (fewer than 8 remaining)
    block_near_end = units[45:50]
    first = 45
    following_end = [{"source": u["text"]}
                     for u in units[first + len(block_near_end):first + len(block_near_end) + REFINE_CONTEXT_AFTER]]
    assert len(following_end) == 0
