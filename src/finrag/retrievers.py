"""Four retrieval setups: bm25, dense, hybrid (RRF), hybrid+rerank."""
from pathlib import Path

import numpy as np

from .fusion import rrf

INDEX_DIR = Path(__file__).resolve().parents[2] / "indexes"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SETUPS = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]


class Retrievers:
    def __init__(self, ids: list[str], texts: list[str], device: str | None = None):
        self.ids, self.texts = ids, texts
        self.device = device
        self._idx = {d: i for i, d in enumerate(ids)}
        self._bm25 = self._emb = self._dense_model = self._reranker = self._stemmer = None

    # ---- BM25 ----
    def bm25(self):
        if self._bm25 is None:
            import bm25s
            import Stemmer
            self._stemmer = Stemmer.Stemmer("english")
            path = INDEX_DIR / "bm25"
            if path.exists():
                self._bm25 = bm25s.BM25.load(str(path))
            else:
                tok = bm25s.tokenize(self.texts, stopwords="en", stemmer=self._stemmer)
                self._bm25 = bm25s.BM25()
                self._bm25.index(tok)
                INDEX_DIR.mkdir(exist_ok=True)
                self._bm25.save(str(path))
        return self._bm25

    def search_bm25(self, query: str, k: int = 100) -> list[str]:
        import bm25s
        b = self.bm25()
        qt = bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer, show_progress=False)
        res, _ = b.retrieve(qt, k=k, show_progress=False)
        return [self.ids[i] for i in res[0]]

    # ---- dense ----
    def dense_model(self):
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(EMB_MODEL, device=self.device)
        return self._dense_model

    def embeddings(self, batch_size: int = 64) -> np.ndarray:
        if self._emb is None:
            path = INDEX_DIR / "dense_bge_small.npy"
            if path.exists():
                self._emb = np.load(path)
            else:
                m = self.dense_model()
                m.max_seq_length = 256
                self._emb = m.encode(self.texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_numpy=True
                                     ).astype("float32")
                INDEX_DIR.mkdir(exist_ok=True)
                np.save(path, self._emb)
        return self._emb

    def search_dense(self, query: str, k: int = 100) -> list[str]:
        q = self.dense_model().encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                      convert_to_numpy=True).astype("float32")[0]
        sims = self.embeddings() @ q
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [self.ids[i] for i in top]

    def search_hybrid(self, query: str, k: int = 100) -> list[str]:
        return rrf([self.search_bm25(query, 100), self.search_dense(query, 100)], top=k)

    # ---- rerank ----
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=256)
        return self._reranker

    def search_rerank(self, query: str, k: int = 100, pool: int = 100) -> list[str]:
        cand = self.search_hybrid(query, pool)
        pairs = [(query, self.texts[self._idx[d]]) for d in cand]
        scores = self.reranker().predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [cand[i] for i in order][:k]

    def search(self, setup: str, query: str, k: int = 100) -> list[str]:
        return {
            "bm25": self.search_bm25,
            "dense": self.search_dense,
            "hybrid_rrf": self.search_hybrid,
            "hybrid_rerank": self.search_rerank,
        }[setup](query, k)

    def passages(self, doc_ids: list[str]) -> list[dict]:
        return [{"id": d, "text": self.texts[self._idx[d]]} for d in doc_ids]
