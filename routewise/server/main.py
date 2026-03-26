"""
RouteWise gateway — main server.

Single entry point: POST /chat
Returns the LLM response plus enough metadata for the log viewer
and cost analysis scripts to do their job.

Startup sequence:
  1. Load the test suite if present and train the ML classifier.
  2. Initialise the cache with the threshold from .env.
  3. Start accepting requests.

The routing model and cache are module-level singletons. This is fine
for a single-process server — if you scale to multiple workers you'd
want them backed by shared storage instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from routing import Router, FAST, CAPABLE
from cache   import Cache
from server.log    import append, get_recent
from server.models import call, FAST_LABEL, CAPABLE_LABEL

app = FastAPI(
    title="RouteWise",
    description="Prompt router — sends each request to the right LLM.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

router = Router()
cache  = Cache(on_event=append)

_SUITE = Path(__file__).parent.parent / "tests" / "test_suite.json"


@app.on_event("startup")
async def _startup():
    if _SUITE.exists():
        try:
            items  = json.loads(_SUITE.read_text())
            texts  = [x["prompt"] for x in items]
            labels = [1 if x["label"] == "complex" else 0 for x in items]
            m = router.train(texts, labels)
            cv = m.get("cv_mean", "n/a")
            print(f"[routewise] classifier trained on {len(texts)} prompts — CV acc {cv}")
        except Exception as e:
            print(f"[routewise] training skipped: {e}")
    else:
        print("[routewise] no test suite found, using rules-only routing")

    print(f"[routewise] cache threshold: {cache.threshold}")
    print("[routewise] ready")


# ---------- request / response schemas ----------

class ChatRequest(BaseModel):
    prompt:     str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=1024, ge=1, le=4096)


class ChatResponse(BaseModel):
    response:        str
    model_used:      str
    model_label:     str
    routing_reason:  str
    confidence:      float
    latency_ms:      float
    routing_ms:      float
    cache_hit:       bool
    cache_score:     float
    in_tokens:       int
    out_tokens:      int
    cost_usd:        float


# ---------- endpoints ----------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t_start = time.perf_counter()
    prompt  = req.prompt.strip()

    if not prompt:
        raise HTTPException(400, "prompt is empty")

    # Step 1 — check cache first
    cached = cache.get(prompt)
    if cached.hit:
        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        append({
            "type":           "request",
            "prompt_snippet": prompt[:120],
            "model_used":     "cached",
            "model_label":    "Cache hit",
            "routing_reason": f"cache hit (score {cached.score:.2f})",
            "confidence":     1.0,
            "latency_ms":     total_ms,
            "routing_ms":     0.0,
            "cache_hit":      True,
            "cache_score":    cached.score,
            "in_tokens":      0,
            "out_tokens":     0,
            "cost_usd":       0.0,
        })
        return ChatResponse(
            response=cached.response,
            model_used="cached", model_label="Cache hit",
            routing_reason=f"cache hit (score {cached.score:.2f})",
            confidence=1.0, latency_ms=total_ms, routing_ms=0.0,
            cache_hit=True, cache_score=cached.score,
            in_tokens=0, out_tokens=0, cost_usd=0.0,
        )

    # Step 2 — route
    route   = router.route(prompt)

    # Step 3 — call LLM
    result  = call(prompt, route.model, max_tokens=req.max_tokens)

    # Step 4 — store in cache (skip on error)
    if not result["error"]:
        cache.put(prompt, result["text"], route.model, route.reasoning)

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    record = {
        "type":           "request",
        "prompt_snippet": prompt[:120],
        "model_used":     route.model,
        "model_label":    result["label"],
        "routing_reason": route.reasoning,
        "confidence":     route.confidence,
        "latency_ms":     total_ms,
        "routing_ms":     route.latency_ms,
        "cache_hit":      False,
        "cache_score":    cached.score,
        "in_tokens":      result["in_tok"],
        "out_tokens":     result["out_tok"],
        "cost_usd":       result["cost"],
        "error":          result["error"],
    }
    append(record)

    return ChatResponse(
        response=result["text"],
        model_used=route.model, model_label=result["label"],
        routing_reason=route.reasoning,
        confidence=route.confidence,
        latency_ms=total_ms, routing_ms=route.latency_ms,
        cache_hit=False, cache_score=cached.score,
        in_tokens=result["in_tok"], out_tokens=result["out_tok"],
        cost_usd=result["cost"],
    )


@app.get("/logs")
async def logs(n: int = 100):
    return {"logs": get_recent(n), "cache": cache.stats()}


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "ml_trained":    router.trained,
        "fast_model":    FAST_LABEL,
        "capable_model": CAPABLE_LABEL,
        "cache":         cache.stats(),
    }


@app.delete("/cache")
async def clear_cache():
    cache.clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
