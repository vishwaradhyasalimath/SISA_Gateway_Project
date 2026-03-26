"""
LLM client for Fast and Capable models.

Uses LiteLLM so we can swap the underlying provider without changing
any other code. Token counting uses tiktoken where possible — it's
more accurate than character-based estimates, which matters when we're
computing cost comparisons.

Pricing constants are approximate free-tier reference values. The goal
is relative comparison (smart routing vs always-capable), so exact
prices matter less than consistency.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

print(f"[debug] ENV PATH: {env_path}")
print(f"[debug] GROQ_API_KEY = '{os.getenv('GROQ_API_KEY', 'NOT FOUND')[:15]}...'")


from litellm import completion
import litellm

litellm.suppress_debug_info = True

FAST_ID    = "groq/llama-3.1-8b-instant"
CAPABLE_ID = "gemini-1.5-flash-preview"

FAST_LABEL    = "Fast model (Groq Llama 3.1 8B)"
CAPABLE_LABEL = "Capable model (Gemini 1.5 Flash)"

_PRICING = {
    "fast":    {"in": 0.05,  "out": 0.08},
    "capable": {"in": 0.075, "out": 0.30},
}


def _get_key(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val or "REPLACE" in val:
        raise EnvironmentError(f"API key {name} is missing or still a placeholder in .env")
    return val


def validate_keys() -> dict:
    status = {}
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        val = os.getenv(name, "").strip()
        if not val:
            status[name] = "missing"
        elif "REPLACE" in val or val.startswith("your_"):
            status[name] = "placeholder"
        else:
            status[name] = "ok"
    return status


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


def estimate_cost(model_key: str, in_tok: int, out_tok: int) -> float:
    r = _PRICING.get(model_key, _PRICING["capable"])
    return round((in_tok / 1_000_000) * r["in"] + (out_tok / 1_000_000) * r["out"], 8)


def call(prompt: str, model_key: str, max_tokens: int = 1024) -> dict:
    is_fast     = (model_key == "fast")
    model_id    = FAST_ID    if is_fast else CAPABLE_ID
    model_label = FAST_LABEL if is_fast else CAPABLE_LABEL

    try:
        api_key = _get_key("GROQ_API_KEY" if is_fast else "GEMINI_API_KEY")
    except EnvironmentError as e:
        return {
            "text": f"[config error: {e}]", "model": model_id,
            "label": model_label, "in_tok": 0, "out_tok": 0,
            "ms": 0.0, "cost": 0.0, "error": str(e),
        }

    t0 = time.perf_counter()
    try:
        resp = completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            api_key=api_key,
        )
        ms   = round((time.perf_counter() - t0) * 1000, 1)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_t  = getattr(usage, "prompt_tokens",     None) or count_tokens(prompt)
        out_t = getattr(usage, "completion_tokens", None) or count_tokens(text)

        return {
            "text": text, "model": model_id, "label": model_label,
            "in_tok": in_t, "out_tok": out_t, "ms": ms,
            "cost": estimate_cost(model_key, in_t, out_t), "error": None,
        }

    except Exception as exc:
        ms       = round((time.perf_counter() - t0) * 1000, 1)
        err_str  = str(exc)
        key_name = "GROQ_API_KEY" if is_fast else "GEMINI_API_KEY"

        if "AuthenticationError" in err_str or "401" in err_str:
            hint = (
                f"Wrong or expired {key_name}. "
                f"Check your .env file. "
                f"Get a fresh key at "
                f"{'console.groq.com' if is_fast else 'aistudio.google.com'}."
            )
        elif "429" in err_str or "RateLimit" in err_str:
            hint = "Rate limit — wait a few seconds and retry."
        else:
            hint = err_str

        return {
            "text": f"[error: {hint}]", "model": model_id, "label": model_label,
            "in_tok": count_tokens(prompt), "out_tok": 0,
            "ms": ms, "cost": 0.0, "error": hint,
        }
