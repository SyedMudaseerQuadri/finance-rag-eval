"""Generate cited answers for 30 FiQA test questions and judge faithfulness with a local LLM.

Needs reports/retrieval_metrics.json (picks the best setup by nDCG@10) and Ollama running.
Writes reports/generation_eval.json and reports/sample_answers.md.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finrag.data import load_corpus, load_qrels, load_queries
from src.finrag.generate import GEN_MODEL, answer_question, call_ollama

REPORTS = ROOT / "reports"
JUDGE_MODEL = "qwen3.5:4b"
N_QUESTIONS = 30
SEED = 0
TOP_K = 5

RUBRIC = (
    "You are a strict fact-checking judge. You get source passages, a question and an answer.\n"
    "Rate whether the ANSWER is faithful to the PASSAGES only (ignore whether it is true in the "
    "real world, and ignore style).\n"
    "- supported: every factual claim in the answer is backed by the passages.\n"
    "- partially: some claims are backed, but others are not found in the passages or are "
    "stretched beyond them.\n"
    "- unsupported: the main claims are not backed by the passages, or the answer contradicts "
    "them.\n"
    "If the answer only says it cannot answer from the passages, rate it supported.\n"
    'Reply with JSON only: {"verdict": "supported|partially|unsupported", "reason": "<one '
    'sentence>"}'
)


def judge(question: str, passages: list[dict], answer: str) -> dict:
    ctx = "\n\n".join(f"[{p['id']}] {p['text'][:1200]}" for p in passages)
    prompt = f"PASSAGES:\n{ctx}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\nJSON verdict:"
    for _ in range(3):
        raw = call_ollama(prompt, model=JUDGE_MODEL, system=RUBRIC, temperature=0.0,
                          think=False, num_predict=200)
        m = re.search(r"\{.*?\}", raw, re.S)
        if m:
            try:
                j = json.loads(m.group(0))
                v = str(j.get("verdict", "")).strip().lower()
                if v in {"supported", "partially", "unsupported"}:
                    return {"verdict": v, "reason": str(j.get("reason", ""))[:300]}
            except json.JSONDecodeError:
                pass
    return {"verdict": "judge_error", "reason": "unparseable judge output"}


def main() -> None:
    metrics = json.loads((REPORTS / "retrieval_metrics.json").read_text())
    best = max(metrics["results"], key=lambda s: metrics["results"][s]["ndcg@10"])
    run = json.loads((REPORTS / "_cache" / f"run_{best}.json").read_text())
    ids, texts = load_corpus()
    text_of = dict(zip(ids, texts))
    queries, qrels = load_queries(), load_qrels("test")
    qids = sorted(q for q in qrels if q in queries)
    sample = random.Random(SEED).sample(qids, N_QUESTIONS)
    print("best retriever:", best)

    rows = []
    for i, q in enumerate(sample, 1):
        passages = [{"id": d, "text": text_of[d]} for d in run[q][:TOP_K]]
        out = answer_question(queries[q], passages, model=GEN_MODEL)
        j = judge(queries[q], passages, out["answer"])
        rows.append({
            "qid": q, "question": queries[q], "answer": out["answer"],
            "retrieved_ids": [p["id"] for p in passages],
            "citation_check": out["validation"],
            "judge": j,
            "gold_in_top5": any(qrels[q].get(p["id"], 0) > 0 for p in passages),
        })
        print(i, j["verdict"], out["validation"]["ok"], flush=True)

    verdicts = Counter(r["judge"]["verdict"] for r in rows)
    n = len(rows)
    summary = {
        "retriever": best,
        "generator_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_questions": n,
        "sample_seed": SEED,
        "top_k_passages": TOP_K,
        "rubric": RUBRIC,
        "caveat": "Judge is a small local 4B model: a noisy proxy for faithfulness. No human "
                  "labels, no inter-judge agreement measured. Treat as a smoke test, not a "
                  "benchmark.",
        "verdict_counts": dict(verdicts),
        "verdict_rates": {k: v / n for k, v in verdicts.items()},
        "citation_validity": {
            "answers_with_all_citations_valid": sum(r["citation_check"]["ok"] for r in rows),
            "answers_uncited": sum(r["citation_check"]["uncited"] for r in rows),
            "answers_with_invalid_citation_ids": sum(bool(r["citation_check"]["invalid"])
                                                     for r in rows),
        },
        "answers_where_a_gold_passage_was_in_top5": sum(r["gold_in_top5"] for r in rows),
        "items": rows,
    }
    (REPORTS / "generation_eval.json").write_text(json.dumps(summary, indent=2))

    md = [f"# Sample answers ({best} top-{TOP_K}, generator {GEN_MODEL})\n",
          "First 5 of the 30 evaluated questions, unedited. Citations are passage IDs from the "
          "retrieved set. Judge verdicts come from a small local model and are noisy.\n"]
    for r in rows[:5]:
        md.append(f"## Q: {r['question']}\n")
        md.append(f"**Answer:** {r['answer']}\n")
        c = r["citation_check"]
        md.append(f"- Cited: {c['cited']} | invalid: {c['invalid']} | validator ok: {c['ok']}")
        md.append(f"- Judge ({JUDGE_MODEL}): **{r['judge']['verdict']}** - {r['judge']['reason']}")
        md.append(f"- Retrieved IDs: {r['retrieved_ids']}\n")
    (REPORTS / "sample_answers.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
