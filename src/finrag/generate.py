"""Grounded answer generation via Ollama with citation validation."""
import os
import re
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:3b")

CITE_RE = re.compile(r"\[([^\[\]\s]+)\]")

SYSTEM = (
    "You answer financial questions using ONLY the passages provided. "
    "Cite every claim with the passage ID in square brackets, e.g. [12345]. "
    "Use only IDs that appear in the passages. If the passages do not contain the answer, "
    "say you cannot answer from the passages. Keep the answer under 120 words."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{p['id']}] {p['text'][:1200]}" for p in passages)
    return f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer (cite passage IDs like [id]):"


def call_ollama(prompt: str, model: str = GEN_MODEL, system: str = SYSTEM,
                temperature: float = 0.1, think: bool | None = None,
                num_predict: int = 400, timeout: int = 600) -> str:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if think is not None:
        body["think"] = think
    last = None
    for attempt in range(5):  # Ollama can 500 when model swaps fight for GPU memory
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.HTTPError as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def validate_citations(answer: str, retrieved_ids: list[str]) -> dict:
    """Flag answers with no citations or with citations outside the retrieved set."""
    cited: list[str] = []
    for c in CITE_RE.findall(answer):
        if c not in cited:
            cited.append(c)
    allowed = set(retrieved_ids)
    valid = [c for c in cited if c in allowed]
    invalid = [c for c in cited if c not in allowed]
    return {
        "cited": cited,
        "valid": valid,
        "invalid": invalid,
        "uncited": len(cited) == 0,
        "ok": len(cited) > 0 and not invalid,
    }


def answer_question(question: str, passages: list[dict], model: str = GEN_MODEL,
                    llm=call_ollama) -> dict:
    ans = llm(build_prompt(question, passages), model=model)
    v = validate_citations(ans, [p["id"] for p in passages])
    return {"answer": ans, "citations": v["valid"], "validation": v}
