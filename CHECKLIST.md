# Done-when checklist

- [x] 1. nDCG@10, Recall@10/100, MRR@10 for all 4 setups on the full 648-query test split (`reports/retrieval_metrics.json`)
- [x] 2. Answers cite passage IDs; validator flags uncited / out-of-set citations (`src/finrag/generate.py`, tested)
- [x] 3. Faithfulness eval on 30 questions, LLM judge with stated rubric and caveat (`reports/generation_eval.json`)
- [x] 4. FastAPI `POST /ask` + `GET /health`; pytest passes (10 tests, LLM mocked); ruff clean
- [x] 5. README numbers copied from `reports/*.json`

Not done (cut for time): UI, pytrec_eval cross-check, domain-tuned reranker, human labels.
Note: hybrid + rerank did not win; dense was best (reported as measured).
