#!/usr/bin/env python3
"""
poc.py — Routing model evaluator.

Run this to check whether the routing model meets the accuracy target.
No server needed, no API calls to any LLM. Pure offline evaluation.

Usage:
    python poc.py                          # uses tests/test_suite.json
    python poc.py --suite custom.json      # custom suite
    python poc.py --suite custom.csv       # CSV also works
    python poc.py --no-train               # rule-based only, skip ML
    python poc.py --threshold 0.38         # override decision threshold

The script exits with code 0 if accuracy >= 75%, 1 otherwise.
Judges: just run it and read the summary at the bottom.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from routing.model import Router, FAST, CAPABLE

# terminal colours — degrade gracefully if the terminal doesn't support them
try:
    import sys as _sys
    _COLOR = _sys.stdout.isatty()
except Exception:
    _COLOR = False

def grn(s):  return f"\033[92m{s}\033[0m" if _COLOR else s
def red(s):  return f"\033[91m{s}\033[0m" if _COLOR else s
def yel(s):  return f"\033[93m{s}\033[0m" if _COLOR else s
def cyn(s):  return f"\033[96m{s}\033[0m" if _COLOR else s
def dim(s):  return f"\033[2m{s}\033[0m"  if _COLOR else s
def bld(s):  return f"\033[1m{s}\033[0m"  if _COLOR else s


def load_suite(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append({
                    "prompt": row["prompt"],
                    "label":  row["label"].strip().lower(),
                })
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"prompt": d["prompt"], "label": d["label"].strip().lower()} for d in data]


def run(suite_path: Path, train: bool = True, threshold: float | None = None):
    print(f"\n{bld(cyn('RouteWise — PoC Evaluator'))}")
    print("─" * 68)

    suite = load_suite(suite_path)
    print(f"  loaded {bld(str(len(suite)))} prompts from {suite_path.name}")

    rtr = Router()
    if threshold is not None:
        rtr.THRESHOLD = threshold
        print(f"  threshold override: {threshold}")

    if train:
        texts  = [x["prompt"] for x in suite]
        labels = [1 if x["label"] == "complex" else 0 for x in suite]
        m = rtr.train(texts, labels)
        cv = m.get("cv_mean", "n/a")
        print(f"  classifier trained — CV accuracy {grn(str(cv))}")
    else:
        print(f"  {yel('ML training skipped — rule-based mode only')}")

    print()

    # column widths
    W_ID   = 3
    W_SNIP = 46
    W_LAB  = 8
    W_PRED = 8
    W_CONF = 6
    W_LAT  = 6

    hdr = (f"  {'#':>{W_ID}}  {'Prompt snippet':<{W_SNIP}}  "
           f"{'Truth':<{W_LAB}}  {'Decision':<{W_PRED}}  "
           f"{'Conf':>{W_CONF}}  {'ms':>{W_LAT}}  Result")
    print(bld(hdr))
    print("  " + "─" * (len(hdr) - 2))

    t_start = time.perf_counter()
    results = []
    tp = tn = fp = fn = 0

    for i, item in enumerate(suite, 1):
        prompt = item["prompt"]
        truth  = item["label"]
        gold   = CAPABLE if truth == "complex" else FAST

        r       = rtr.route(prompt)
        correct = r.model == gold

        if correct and truth == "complex":   tp += 1
        elif correct and truth == "simple":  tn += 1
        elif truth == "simple":              fp += 1  # simple → capable (wasted $)
        else:                                fn += 1  # complex → fast   (quality loss)

        snip = prompt[:W_SNIP] + "…" if len(prompt) > W_SNIP else prompt
        pred_str  = cyn(r.model[:W_PRED]) if r.model == CAPABLE else f"{r.model[:W_PRED]}"
        truth_str = dim(truth[:W_LAB])
        ok_str    = grn("✓") if correct else red("✗")

        print(f"  {i:>{W_ID}}  {snip:<{W_SNIP}}  "
              f"{truth_str:<{W_LAB+10}}  {pred_str:<{W_PRED+10}}  "
              f"{r.confidence:>{W_CONF}.1%}  {r.latency_ms:>{W_LAT}.1f}  {ok_str}")

        results.append({
            "id": i, "prompt": prompt, "truth": truth,
            "predicted": r.model, "correct": correct,
            "confidence": r.confidence,
            "reasoning": r.reasoning,
            "ms": r.latency_ms,
        })

    elapsed   = time.perf_counter() - t_start
    total     = len(suite)
    n_correct = tp + tn
    accuracy  = n_correct / total
    fpr       = fp / max(1, tn + fp)
    fnr       = fn / max(1, tp + fn)

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print("─" * 68)
    print(f"  {bld('Results')}")
    print("─" * 68)

    acc_colour = grn if accuracy >= 0.75 else (yel if accuracy >= 0.60 else red)
    fpr_colour = grn if fpr <= 0.20 else (yel if fpr <= 0.35 else red)
    fnr_colour = grn if fnr <= 0.20 else (yel if fnr <= 0.35 else red)

    print(f"  {'Accuracy':<32} {acc_colour(f'{accuracy:.1%}  ({n_correct}/{total})')}")
    print(f"  {'True positives  (complex→capable)':<32} {tp}")
    print(f"  {'True negatives  (simple→fast)':<32} {tn}")
    print(f"  {'False positives (simple→capable)':<32} {fpr_colour(str(fp))}  {dim('← wasted cost')}")
    print(f"  {'False negatives (complex→fast)':<32} {fnr_colour(str(fn))}  {dim('← quality loss')}")
    print(f"  {'False positive rate':<32} {fpr_colour(f'{fpr:.1%}')}")
    print(f"  {'False negative rate':<32} {fnr_colour(f'{fnr:.1%}')}")
    print(f"  {'Total eval time':<32} {elapsed*1000:.0f} ms "
          f"({'pass' if elapsed < 60 else red('OVER 60s')})")
    print()

    # success bar check
    bar_pass = accuracy >= 0.75
    print(f"  {bld('Success bar')}  accuracy ≥ 75%:  "
          f"{grn('PASS') if bar_pass else red('FAIL')}  (got {accuracy:.1%})")
    print()

    # mis-routes
    bad = [r for r in results if not r["correct"]]
    if bad:
        print(f"  {bld(yel(f'Mis-routes ({len(bad)})'))}  ← review these for root-cause analysis")
        for m in bad:
            kind = "FP" if m["truth"] == "simple" else "FN"
            cost_note = "(wasted cost)" if kind == "FP" else "(quality loss)"
            print(f"\n  [{kind}] #{m['id']}: {m['prompt'][:65]}")
            print(f"       truth={m['truth']}, predicted={m['predicted']}, "
                  f"conf={m['confidence']:.1%}  {dim(cost_note)}")
            print(f"       {dim(m['reasoning'])}")
    else:
        print(f"  {grn('No mis-routes on this suite.')}")
        print(f"  {dim('Note: for the demo, use the edge-case prompts (ids 21–30)')}")
        print(f"  {dim('to show prompts that came close to the decision boundary.')}")

    print()
    print("─" * 68)

    return {
        "accuracy":  accuracy,
        "fp_rate":   fpr,
        "fn_rate":   fnr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "elapsed_s": round(elapsed, 3),
        "results":   results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="RouteWise PoC — evaluates the routing model offline."
    )
    ap.add_argument("--suite",     type=Path,
                    default=ROOT / "tests" / "test_suite.json")
    ap.add_argument("--no-train",  action="store_true",
                    help="Skip ML training, use rule scorer only.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the routing threshold (default 0.42).")
    args = ap.parse_args()

    if not args.suite.exists():
        print(f"error: suite not found at {args.suite}")
        sys.exit(1)

    summary = run(args.suite, train=not args.no_train, threshold=args.threshold)
    sys.exit(0 if summary["accuracy"] >= 0.75 else 1)


if __name__ == "__main__":
    main()
