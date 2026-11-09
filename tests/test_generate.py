from src.finrag.generate import answer_question, build_prompt, validate_citations

P = [{"id": "11", "text": "Index funds are cheap."}, {"id": "22", "text": "Bonds pay coupons."}]


def test_validator_flags():
    assert validate_citations("Cheap [11].", ["11", "22"])["ok"]
    v = validate_citations("Made up [99] and [11].", ["11", "22"])
    assert not v["ok"] and v["invalid"] == ["99"]
    v = validate_citations("No citations here.", ["11"])
    assert v["uncited"] and not v["ok"]


def test_prompt_contains_ids():
    assert "[11]" in build_prompt("q?", P)


def test_answer_question_mocked():
    out = answer_question("why?", P, llm=lambda prompt, model=None: "Because [22].")
    assert out["citations"] == ["22"] and out["validation"]["ok"]
