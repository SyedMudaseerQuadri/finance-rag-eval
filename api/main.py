"""FastAPI service: POST /ask, GET /health."""
import os
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.finrag.generate import GEN_MODEL, answer_question, call_ollama

SETUP = os.environ.get("RETRIEVER", "dense")

app = FastAPI(title="finance-rag-eval", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def _real_retriever():
    from src.finrag.data import load_corpus
    from src.finrag.retrievers import Retrievers
    ids, texts = load_corpus()
    return Retrievers(ids, texts)


def get_state() -> dict:
    return {"retriever": _real_retriever(), "llm": call_ollama, "setup": SETUP}


@app.get("/health")
def health():
    return {"status": "ok", "retriever": SETUP, "generator": GEN_MODEL}


@app.post("/ask")
def ask(req: AskRequest, state: dict = Depends(get_state)):
    ret = state["retriever"]
    ids = ret.search(state["setup"], req.question, req.k)
    passages = ret.passages(ids)
    out = answer_question(req.question, passages, llm=state["llm"])
    return {
        "answer": out["answer"],
        "citations": out["citations"],
        "passages": passages,
        "retriever": state["setup"],
        "validation": out["validation"],
    }
