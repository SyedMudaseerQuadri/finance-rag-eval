"""Evaluate the 4 retrieval setups on the full FiQA test split.

Writes reports/retrieval_metrics.json, reports/retrieval_metrics.png and
reports/_cache/runs_<setup>.json (top-100 per query, reused by the generation eval).
"""
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.finrag.data import load_corpus, load_qrels, load_queries
from src.finrag.metrics import evaluate
from src.finrag.retrievers import EMB_MODEL, RERANK_MODEL, SETUPS, Retrievers

REPORTS = ROOT / "reports"


def encode_with_fallback(r: Retrievers) -> None:
    import torch
    try:
        r.embeddings(batch_size=64)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM, retrying with batch 16")
        torch.cuda.empty_cache()
        try:
            r.embeddings(batch_size=16)
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM again, falling back to CPU")
            r._dense_model = None
            r.device = "cpu"
            r.embeddings(batch_size=32)


def main() -> None:
    (REPORTS / "_cache").mkdir(parents=True, exist_ok=True)
    ids, texts = load_corpus()
    queries, qrels = load_queries(), load_qrels("test")
    qids = [q for q in qrels if q in queries]
    print(f"corpus={len(ids)} test_queries={len(qids)}")
    r = Retrievers(ids, texts)
    r.bm25()
    t0 = time.time()
    encode_with_fallback(r)
    print(f"index ready, dense encode/load took {time.time() - t0:.0f}s")
    r.reranker()
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "cpu"

    results = {}
    for setup in SETUPS:
        run, lat = {}, []
        r.search(setup, queries[qids[0]], 100)  # warm-up
        for q in qids:
            t = time.perf_counter()
            run[q] = r.search(setup, queries[q], 100)
            lat.append((time.perf_counter() - t) * 1000)
        m = evaluate(run, qrels)
        m["latency_ms_mean"] = statistics.mean(lat)
        m["latency_ms_median"] = statistics.median(lat)
        m["latency_ms_p95"] = sorted(lat)[int(0.95 * len(lat)) - 1]
        results[setup] = m
        (REPORTS / "_cache" / f"run_{setup}.json").write_text(json.dumps(run))
        print(setup, {k: round(v, 4) for k, v in m.items()})

    out = {
        "dataset": "BEIR FiQA-2018 test split",
        "corpus_size": len(ids),
        "n_test_queries": len(qids),
        "embedding_model": EMB_MODEL,
        "reranker_model": RERANK_MODEL,
        "rrf_k": 60,
        "rerank_pool": 100,
        "hardware": {"device": gpu, "python": platform.python_version()},
        "latency_note": "per-query wall-clock incl. query encoding / reranking, "
                        "after warm-up, single query at a time, shared GPU",
        "results": results,
    }
    (REPORTS / "retrieval_metrics.json").write_text(json.dumps(out, indent=2))

    metrics = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    colors = ["#8a8f98", "#3b82f6", "#10b981", "#f59e0b"]
    for i, s in enumerate(SETUPS):
        vals = [results[s][m] for m in metrics]
        bars = ax.bar([x + (i - 1.5) * w for x in range(len(metrics))], vals, w,
                      label=s, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title(f"FiQA-2018 test ({len(qids)} queries): 4 retrieval setups")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "retrieval_metrics.png", dpi=150)


if __name__ == "__main__":
    main()
