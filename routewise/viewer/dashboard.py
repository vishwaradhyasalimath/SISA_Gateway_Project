"""
Log viewer dashboard.

Keep this simple — it exists so judges can inspect routing decisions
during the demo without digging through a JSON file. Not a product.

Run:
    streamlit run viewer/dashboard.py
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import requests as _req
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

GATEWAY = "http://localhost:8000"
LOG_FILE = ROOT / "logs" / "requests.jsonl"

st.set_page_config(
    page_title="RouteWise logs",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.badge-fast    { background:#DBEAFE; color:#1E40AF; padding:2px 9px;
                 border-radius:10px; font-size:0.78rem; font-weight:600; }
.badge-capable { background:#D1FAE5; color:#065F46; padding:2px 9px;
                 border-radius:10px; font-size:0.78rem; font-weight:600; }
.badge-cached  { background:#FEF3C7; color:#92400E; padding:2px 9px;
                 border-radius:10px; font-size:0.78rem; font-weight:600; }
.dim           { color:#9CA3AF; font-size:0.8rem; font-style:italic; }
hr.thin        { border:none; border-top:1px solid #F3F4F6; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


def badge(model: str) -> str:
    cls = {"fast": "badge-fast", "capable": "badge-capable"}.get(model, "badge-cached")
    label = {"fast": "⚡ Fast", "capable": "🧠 Capable"}.get(model, "💾 Cached")
    return f'<span class="{cls}">{label}</span>'


def ts_fmt(ts) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "—"


@st.cache_data(ttl=3)
def fetch_from_api(n):
    if not _REQUESTS_OK:
        return [], {}, False
    try:
        r = _req.get(f"{GATEWAY}/logs?n={n}", timeout=2)
        d = r.json()
        return d.get("logs", []), d.get("cache", {}), True
    except Exception:
        return [], {}, False


def fetch_from_file(n) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    lines = []
    with open(LOG_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(lines[-n:]))


@st.cache_data(ttl=3)
def fetch_health():
    if not _REQUESTS_OK:
        return {}, False
    try:
        r = _req.get(f"{GATEWAY}/health", timeout=2)
        return r.json(), True
    except Exception:
        return {}, False


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    n_rows       = st.slider("Rows to show", 20, 200, 100, 10)
    auto_refresh = st.checkbox("Auto-refresh (3s)")
    if st.button("↺  Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔌 Gateway")
    health, gw_ok = fetch_health()
    if gw_ok:
        st.success("Online")
        st.caption(f"Fast: {health.get('fast_model','?')}")
        st.caption(f"Capable: {health.get('capable_model','?')}")
        st.caption(f"ML trained: {'yes' if health.get('ml_trained') else 'no'}")
    else:
        st.error("Offline — showing file logs")

    st.markdown("---")
    st.markdown("### 🧪 Test a prompt")
    test_prompt = st.text_area("", height=70, placeholder="Type something...")
    if st.button("Send") and test_prompt.strip():
        if gw_ok:
            with st.spinner("routing..."):
                try:
                    r = _req.post(f"{GATEWAY}/chat",
                                  json={"prompt": test_prompt.strip()},
                                  timeout=30)
                    d = r.json()
                    st.info(f"**{d.get('model_label','?')}**")
                    st.caption(d.get("routing_reason", ""))
                    ans = d.get("response", "")
                    st.write(ans[:300] + ("…" if len(ans) > 300 else ""))
                    st.cache_data.clear()
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Gateway is offline.")


# ── main panel ────────────────────────────────────────────────────────────────

st.markdown("## ⚡ RouteWise — Request Log")

logs, cache_stats, from_api = fetch_from_api(n_rows)
if not from_api:
    logs = fetch_from_file(n_rows)
    cache_stats = {}

req_logs = [l for l in logs if "prompt_snippet" in l]

# metric cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
total      = len(req_logs)
fast_n     = sum(1 for l in req_logs if l.get("model_used") == "fast")
capable_n  = sum(1 for l in req_logs if l.get("model_used") == "capable")
cached_n   = sum(1 for l in req_logs if l.get("cache_hit") or l.get("model_used") == "cached")
lats       = [l["latency_ms"] for l in req_logs if l.get("latency_ms")]
avg_lat    = round(sum(lats) / len(lats), 0) if lats else 0
total_cost = sum(l.get("cost_usd", 0) for l in req_logs)

for col, val, lbl in [
    (c1, total,             "Requests"),
    (c2, fast_n,            "Fast model"),
    (c3, capable_n,         "Capable model"),
    (c4, cached_n,          "Cache hits"),
    (c5, f"{avg_lat:.0f}ms","Avg latency"),
    (c6, f"${total_cost:.6f}", "Est. cost"),
]:
    col.metric(lbl, val)

# cache stats
if cache_stats:
    with st.expander("📦 Cache stats"):
        cols = st.columns(4)
        for col, k, lbl in [
            (cols[0], "lookups",   "Lookups"),
            (cols[1], "hits",      "Hits"),
            (cols[2], "hit_rate",  "Hit rate %"),
            (cols[3], "threshold", "Threshold"),
        ]:
            col.metric(lbl, cache_stats.get(k, "—"))

st.markdown("---")

# filters
fc1, fc2, fc3 = st.columns([2, 3, 2])
model_f  = fc1.selectbox("Model", ["All", "fast", "capable", "cached"])
search   = fc2.text_input("Search", placeholder="keyword in prompt...")
sort_by  = fc3.selectbox("Sort", ["Newest first", "Slowest first", "Most expensive"])

filtered = req_logs.copy()
if model_f != "All":
    filtered = [l for l in filtered
                if l.get("model_used") == model_f
                or (model_f == "cached" and l.get("cache_hit"))]
if search:
    filtered = [l for l in filtered
                if search.lower() in l.get("prompt_snippet", "").lower()]
if sort_by == "Slowest first":
    filtered.sort(key=lambda l: l.get("latency_ms", 0), reverse=True)
elif sort_by == "Most expensive":
    filtered.sort(key=lambda l: l.get("cost_usd", 0), reverse=True)

st.caption(f"{len(filtered)} of {total} entries")

if not filtered:
    st.info("No entries yet — send some prompts to the gateway.")
else:
    # table header
    hc = st.columns([1, 5, 2, 3, 2, 2])
    for col, lbl in zip(hc, ["Time", "Prompt", "Model", "Routing reason", "Latency", "Cache"]):
        col.markdown(f"**{lbl}**")
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    for log in filtered[:n_rows]:
        cols = st.columns([1, 5, 2, 3, 2, 2])
        cols[0].caption(ts_fmt(log.get("ts")))

        snip = log.get("prompt_snippet", "—")
        cols[1].markdown(f'<span style="font-size:0.9rem">{snip[:80]}</span>',
                         unsafe_allow_html=True)

        cols[2].markdown(badge(log.get("model_used", "?")), unsafe_allow_html=True)

        reason = log.get("routing_reason", "—")
        cols[3].markdown(f'<span class="dim">{reason[:55]}{"…" if len(reason)>55 else ""}</span>',
                         unsafe_allow_html=True)

        ms = log.get("latency_ms", 0)
        dot = "🟢" if ms < 500 else ("🟡" if ms < 2000 else "🔴")
        cols[4].caption(f"{dot} {ms:.0f} ms")
        cols[5].caption("✅ Hit" if log.get("cache_hit") else "❌ Miss")

# cost comparison
with st.expander("💰 Cost comparison — smart routing vs always-Capable"):
    non_cached = [l for l in req_logs
                  if not l.get("cache_hit") and l.get("model_used") != "cached"]
    if non_cached:
        smart_cost = sum(l.get("cost_usd", 0) for l in non_cached)
        # Hypothetical always-capable cost for the same token counts
        always_cap = sum(
            l.get("in_tokens", 0) * 0.075 / 1e6 +
            l.get("out_tokens", 0) * 0.30 / 1e6
            for l in non_cached
        )
        saving     = always_cap - smart_cost
        pct        = saving / always_cap * 100 if always_cap > 0 else 0

        st.metric("Always-Capable cost", f"${always_cap:.6f}")
        st.metric("Smart routing cost",  f"${smart_cost:.6f}")
        st.metric("Saving",              f"${saving:.6f}",
                  delta=f"{pct:.1f}% reduction",
                  delta_color="normal")
        if pct >= 25:
            st.success("Success bar (>25% saving): PASS")
        else:
            st.warning(f"Success bar: not yet — {pct:.1f}% saving so far")
    else:
        st.caption("No non-cached requests yet.")

if auto_refresh:
    time.sleep(3)
    st.cache_data.clear()
    st.rerun()
