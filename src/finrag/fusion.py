def rrf(rankings: list[list[str]], k: int = 60, top: int = 100) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum 1/(k + rank)."""
    scores: dict[str, float] = {}
    for r in rankings:
        for i, d in enumerate(r):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + i + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))][:top]
