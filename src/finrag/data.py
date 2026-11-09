"""Loading BEIR-format data."""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fiqa"


def load_corpus(path: Path = DATA_DIR) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts); title is prepended to text when present."""
    ids, texts = [], []
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(str(d["_id"]))
            texts.append((d.get("title", "") + " " + d["text"]).strip())
    return ids, texts


def load_queries(path: Path = DATA_DIR) -> dict[str, str]:
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        return {str(d["_id"]): d["text"] for d in map(json.loads, f)}


def load_qrels(split: str = "test", path: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for q, d, s in r:
            qrels.setdefault(str(q), {})[str(d)] = int(s)
    return qrels
