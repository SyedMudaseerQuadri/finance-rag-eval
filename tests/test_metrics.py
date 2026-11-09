import math

from src.finrag.fusion import rrf
from src.finrag.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_zero():
    q = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], q, 10) == 1.0
    assert ndcg_at_k(["x", "y"], q, 10) == 0.0


def test_ndcg_known_value():
    q = {"a": 1}
    # relevant doc at rank 3 -> dcg = 1/log2(4) = 0.5, idcg = 1
    assert math.isclose(ndcg_at_k(["x", "y", "a"], q, 10), 0.5)


def test_recall_and_mrr():
    q = {"a": 1, "b": 1, "z": 0}
    assert recall_at_k(["a", "x", "b"], q, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], q, 3) == 1.0
    assert mrr_at_k(["x", "b"], q, 10) == 0.5
    assert mrr_at_k(["x", "y"], q, 10) == 0.0
    assert mrr_at_k(["x", "y", "a"], q, 2) == 0.0


def test_rrf_prefers_docs_in_both_lists():
    out = rrf([["a", "b", "c"], ["c", "a", "d"]], top=4)
    assert out[0] == "a" and set(out) == {"a", "b", "c", "d"}
