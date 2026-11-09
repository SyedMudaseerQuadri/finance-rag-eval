from fastapi.testclient import TestClient

from api.main import app, get_state


class FakeRet:
    def search(self, setup, query, k=100):
        return ["1", "2", "3"][:k]

    def passages(self, ids):
        return [{"id": i, "text": f"text {i}"} for i in ids]


def fake_llm(prompt, model=None):
    return "Answer based on [1] and [99]."


def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"


def test_ask_mocked():
    app.dependency_overrides[get_state] = lambda: {
        "retriever": FakeRet(), "llm": fake_llm, "setup": "hybrid_rerank"}
    c = TestClient(app)
    r = c.post("/ask", json={"question": "what is x?", "k": 2})
    assert r.status_code == 200
    j = r.json()
    assert j["retriever"] == "hybrid_rerank"
    assert [p["id"] for p in j["passages"]] == ["1", "2"]
    assert j["citations"] == ["1"]
    assert j["validation"]["invalid"] == ["99"]
    app.dependency_overrides.clear()


def test_ask_validation():
    # mock the retriever so the test never needs the downloaded corpus
    app.dependency_overrides[get_state] = lambda: {
        "retriever": FakeRet(), "llm": fake_llm, "setup": "dense"}
    c = TestClient(app)
    assert c.post("/ask", json={"question": "", "k": 5}).status_code == 422
    app.dependency_overrides.clear()
