"""Own IR metrics (linear-gain nDCG, matching pytrec_eval ndcg_cut for binary qrels)."""
import math


def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [max(qrel.get(d, 0), 0) for d in ranked[:k]]
    ideal = sorted((max(v, 0) for v in qrel.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, v in qrel.items() if v > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def mrr_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    qs = [q for q in qrels if q in run]
    n = len(qs)
    return {
        "ndcg@10": sum(ndcg_at_k(run[q], qrels[q], 10) for q in qs) / n,
        "recall@10": sum(recall_at_k(run[q], qrels[q], 10) for q in qs) / n,
        "recall@100": sum(recall_at_k(run[q], qrels[q], 100) for q in qs) / n,
        "mrr@10": sum(mrr_at_k(run[q], qrels[q], 10) for q in qs) / n,
        "n_queries": n,
    }
