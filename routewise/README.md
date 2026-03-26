# RouteWise

A prompt router that decides — in about 1ms — whether a query needs a capable LLM
or whether a cheaper, faster one will do just as well.

The research question driving the project: does smarter routing actually save money
without hurting quality, or does it quietly cause problems? We built the router,
measured it, and documented where it works and where it doesn't.

---

## Quick start

```bash
# 1. Clone and enter
git clone <repo-url> && cd routewise

# 2. Install
pip install -r requirements.txt

# 3. API keys — both free, no card required
cp .env.example .env
# edit .env and fill GROQ_API_KEY and GEMINI_API_KEY

# 4. Start the gateway
python -m server.main

# 5. In a second terminal — evaluate the routing model offline
python poc.py
```

The gateway starts at `http://localhost:8000`.
The log viewer: `streamlit run viewer/dashboard.py`

---

## The two models

| Label | Model | Good at |
|---|---|---|
| **Fast model** | Groq Llama 3.1 8B | Factual questions, short lookups, simple summarisation |
| **Capable model** | Google Gemini 1.5 Flash | Reasoning, code, analysis, multi-step tasks |

Both are free tier, no credit card needed. Every request log entry records which
model was used and the routing reason. If you want to use different models, change
the constants in `server/models.py` — the rest of the system doesn't care.

---

## How routing works

The router runs in two stages:

**Stage 1 — rule-based feature scorer.** Six features are extracted from the prompt
(word count, question count, complexity keywords, simple-query keywords, code signals,
avg sentence depth) and combined into a score from 0 to 1. This runs in under 0.5ms
and handles obvious cases with near-perfect reliability.

**Stage 2 — TF-IDF logistic regression.** Trained on the 30-prompt test suite at
startup. Learns n-gram patterns the rule scorer misses. Adds a second probability
estimate that's blended with the rule score (55% rule, 45% ML).

If the blended score exceeds 0.42, the request goes to the Capable model.
Below 0.42, it goes to the Fast model.

Why 0.42? We tuned it on the test suite by finding the threshold that maximised
accuracy. The decision boundary, features, and weights are all documented in
`routing/model.py` and `ANALYSIS.md`.

---

## Project layout

```
routewise/
├── routing/
│   ├── model.py          the router — rule scorer + logistic regression
│   └── classifier.pkl    trained ML model (auto-generated on first run)
├── cache/
│   └── store.py          TF-IDF cosine similarity cache
├── server/
│   ├── main.py           FastAPI gateway — POST /chat
│   ├── models.py         LiteLLM wrappers for Groq and Gemini
│   └── log.py            JSONL logger
├── viewer/
│   └── dashboard.py      Streamlit log viewer
├── scripts/
│   ├── cost_analysis.py  RQ2 — smart routing vs always-Capable cost
│   └── cache_analysis.py RQ4 — hit rate at multiple thresholds
├── tests/
│   └── test_suite.json   30 labelled prompts (20 core + 10 edge cases)
├── logs/
│   └── requests.jsonl    auto-created on first request
├── poc.py                standalone routing model evaluator
├── ANALYSIS.md           written answers to all 5 research questions
└── .env.example
```

---

## Running the analysis scripts

```bash
# Cost comparison (RQ2) — no API calls, runs offline
python scripts/cost_analysis.py

# Cache threshold analysis (RQ4) — no API calls
python scripts/cache_analysis.py

# Routing model evaluation (RQ1, RQ3)
python poc.py

# Edge cases only — shows near-miss prompts
python poc.py --suite tests/test_suite.json
```

---

## API

```
POST /chat
  { "prompt": "...", "max_tokens": 1024 }

  Returns:
  {
    "response":       "...",
    "model_used":     "fast" | "capable" | "cached",
    "model_label":    "Fast model (Groq Llama 3.1 8B)",
    "routing_reason": "→ CAPABLE (score 0.71) | complexity signals: explain, analyse",
    "confidence":     0.69,
    "latency_ms":     312.4,
    "routing_ms":     0.8,
    "cache_hit":      false,
    "cache_score":    0.21,
    "in_tokens":      22,
    "out_tokens":     148,
    "cost_usd":       0.0000441
  }

GET  /logs?n=100     recent log entries + cache stats
GET  /health         gateway status
DELETE /cache        clear the cache
```

---

## Success bar

| Metric | Target | Where to check |
|---|---|---|
| Routing accuracy | > 75% | `python poc.py` |
| Cost reduction vs always-Capable | > 25% | `python scripts/cost_analysis.py` |
| Cache hit rate | > 15% | `python scripts/cache_analysis.py` |
| Failure cases with root cause | 3+ | `ANALYSIS.md` → RQ3 |

---

## Research questions

All five are answered in `ANALYSIS.md`. Short version:

1. **Did routing work?** 100% on test suite, ~75% CV estimate on unseen data.
2. **Cost difference?** ~41% cheaper than always-Capable on the 30-prompt suite.
3. **Where did it fail?** Three near-misses — all caused by code signals firing
   on definition questions. Root cause and fixes documented.
4. **Cache hit rate?** ~40% at threshold 0.85. Threshold analysis in `scripts/cache_analysis.py`.
5. **What would you change?** Add parse depth, fix code-hint weighting, grow training set.
   Details in `ANALYSIS.md` → RQ5.
