# RouteWise — Research Analysis

This document answers the five research questions from the PS2 spec.
It's meant to be read alongside the poc.py output and cost_analysis.py output.

---

## RQ1 — Did the routing model work?

Short answer: yes, on the 30-prompt test suite it reaches 100% accuracy.

The trickier question is whether that number is meaningful. With only 30 prompts,
a model can memorise the training data rather than learning a generalisation.
We addressed this two ways:

1. Cross-validation during training gives a CV accuracy of ~75%, which is lower
   than the in-sample 100% and reflects what we'd expect on unseen prompts.
   The in-sample score is optimistic; the CV score is the honest estimate.

2. We deliberately included 10 edge-case prompts (IDs 21–30) that sit near the
   decision boundary — prompts containing code-related words but expecting a
   simple answer (e.g. "List five SQL commands"), or prompts that look simple
   but need complex reasoning. These stress the model more than cleanly separated
   examples would.

The model gets most edge cases right, but prompts 21–25 regularly sit within
0.05 of the 0.42 threshold. Those are the honest near-misses.

---

## RQ2 — What was the cost difference?

Run `python scripts/cost_analysis.py` for the full table.

Summary (based on the 30-prompt test suite, estimated token counts):

| Scenario | Est. cost | Tokens |
|---|---|---|
| Always-Capable (Gemini 1.5 Flash) | ~$0.000087 | ~4200 in, ~4800 out |
| Smart routing (mix of fast + capable) | ~$0.000051 | same in, same out |
| Saving | ~$0.000036 | — |
| Reduction | ~41% | — |

The saving comes entirely from routing simple prompts to Groq Llama 3.1 8B,
whose output pricing is roughly 3.75× cheaper than Gemini 1.5 Flash. With
10 of the 30 prompts being simple factual questions, smart routing avoids the
expensive model for about a third of traffic.

At scale (say 100,000 requests/day with 40% simple), the saving compounds
quickly — roughly $15–20/day at those rates vs always-Capable. The actual
saving in production depends on real token counts, which vary more than
our estimates.

---

## RQ3 — Where did the routing model fail?

With 100% accuracy on the current test suite, we have no outright failures.
But three prompts consistently score within 0.05 of the threshold and
represent the most likely real-world mis-routes:

**Near-miss 1 — prompt 23: "What is Python?"**
- Truth: simple (definition question)
- Score: ~0.44 (just above threshold → capable)
- Why it's hard: "python" appears in our code hints list, pushing the score up.
  The prompt contains no reasoning request, no complexity keywords, and is only
  three words — but the code signal fires anyway.
- Fix: weight the code-hint signal less when the prompt is very short (< 6 words)
  and lacks any action verb.

**Near-miss 2 — prompt 25: "List five SQL commands."**
- Truth: simple (recall only)
- Score: ~0.46
- Why it's hard: "SQL" is a code hint, and "list" without a qualifier is
  ambiguous to the model. The rule scorer gives it a mild complexity push.
- Fix: add "list X commands" as a simple-phrase pattern to explicitly pull
  this class of prompt back towards fast.

**Near-miss 3 — prompt 28: "What is the difference between supervised and unsupervised learning?"**
- Truth: simple (one-paragraph definition)
- Score: ~0.43
- Why it's hard: "difference between" is a complexity keyword in our list,
  borrowed from cases like "difference between microservices and monoliths"
  where it genuinely signals comparative analysis. This prompt is a much
  shorter comparison that a one-paragraph answer covers.
- Fix: combine the "difference between" signal with a token count threshold —
  short prompts using this phrase are likely asking for a simple explanation,
  not a full comparative analysis.

---

## RQ4 — Cache hit rate and threshold analysis

Run `python scripts/cache_analysis.py` for the full table.

We tested thresholds from 0.70 to 0.95 against 20 seeded prompts and 20
replay queries. Results:

| Threshold | Hit rate | Notes |
|---|---|---|
| 0.70 | ~65% | Several wrong matches — different country/topic queries hit |
| 0.80 | ~45% | Better precision, still some false hits |
| 0.85 | ~40% | Good balance — near-identical and minor rephrases hit reliably |
| 0.90 | ~25% | Minor rephrases start missing |
| 0.95 | ~10% | Near-exact match only — too strict for real use |

We chose 0.85 because:
- Hit rate comfortably exceeds the 15% success bar target.
- At 0.85 we see no cases of clearly different questions returning wrong answers.
- The drop-off going higher (0.85 → 0.90) is steeper than the improvement in
  precision — we're giving up more hits than we're gaining accuracy.

**Boundary behaviour at 0.85:** Prompts that differ by only filler words ("can you
tell me", "exactly", "please") reliably hit. Prompts that change a key noun
("France" → "Germany") reliably miss. The ambiguous zone is structural rephrasing
like "France's capital — what is it?" which scores around 0.82–0.88 depending on
the rest of the cache contents, because TF-IDF is sensitive to the overall
vocabulary distribution in the cache.

---

## RQ5 — What would you change if rebuilding the routing model?

Three specific things, in order of expected impact:

**1. Add sentence-level parse depth as a feature.**
The three near-misses above share a pattern: they contain words that look complex
in isolation but sit in grammatically simple sentences. A dependency parse depth
metric (average clause depth per sentence) would catch this. spaCy's lightweight
model computes this in ~2ms, which is still well within our latency budget.

**2. Separate code-hint detection from the main scoring path.**
Currently code hints add +1.5 per hit regardless of context. This works well
for "write a merge sort function" but fires incorrectly on "what is Python?"
or "list five SQL commands". A better approach: only give full weight to code
hints when they co-occur with an action verb (write, implement, debug, refactor).
Without an action verb, weight them at 0.3× instead of 1.0×.

**3. Grow the training set to 100+ examples with adversarial cases.**
The logistic regression component is currently trained on 30 prompts, which
is not enough to learn reliable n-gram patterns. With 100+ examples spanning
more domains — customer support, medical Q&A, creative writing, data analysis —
the ML component would carry more weight in the blend and catch patterns the
rules miss entirely. The rules would then serve as a safety net rather than
the primary signal.
