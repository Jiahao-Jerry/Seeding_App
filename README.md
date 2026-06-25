# Style-SAE & Rewrite-Audit — design and build notes

## Pipeline Overview

**Input → Output in 6 steps:**

```
phase3_curation/out/curated_corpus.parquet   (9,500 posts, 19 topics)
        │
        ▼
[1] dataset.py          slice BGE-M3 embeddings by row_idx
        │                → data/sae2/dataset.parquet  +  bge.npy
        ▼
[2] extract_qwen.py     run Qwen2.5-7B on Colab, extract hidden states at layers 14/18/22
        │                → data/sae2/qwen/qwen_L{14,18,22}.npy
        ▼
[3] representations.py  build 8 variants: {bge, qwen@layer} × {raw, kNN-residual}
        │                → data/sae2/variants/*.npy   (kNN removes ~80% topic signal)
        ▼
[4] label_subset.py     score 2,000 stratified posts on 9 style axes via Claude Haiku
        │                → data/sae2/axis_labels.parquet
        ▼
[5] layer_sweep.py      train one SAE per Qwen layer, score axis alignment, pick winner
        │                → data/sae2/variants/qwen{L}_knn/  +  data/sae2/report.md
        │                  Result: L22 best (all 9 axes confirmed)
        ▼
[6] correlate.py        per-feature Pearson r + axis lift against labels → interpret.py
```

**9 style axes** (single source of truth: `config/axes.py`):
`reading_level · concreteness · narrativity · hedging · tone · warmth · self_disclosure · casualness · humor`

---

This is the working brief for the next chunk of the delivery project. It's written
to be read by whoever implements it *and* by the rest of us, so it spends as much
time on **why** as on **what**. Read it once top to bottom before writing code —
several of the design choices only make sense in light of each other.

---

## 1. What we're actually trying to do

The product is a feed that adapts **how** a post is written to a reader's taste,
while leaving **what** it's about untouched. An LLM rewrites the *delivery* of a
post — more or less formal, playful, personal, story-like, hedged, and so on. The
delivery dimensions we care about are defined once, in `seeding_app/config/axes.py`
(nine axes: `reading_level, concreteness, narrativity, hedging, tone, warmth,
self_disclosure, casualness, humor`). That file is the single source of truth;
read the axis list and metadata from it, never hardcode dimension names anywhere
else.

Two things are missing that this work builds:

1. **A reproducible numerical fingerprint of a post's style.** We want a small,
   fixed vector of "style coordinates" for any post that is computed deterministically
   from a neural representation — *not* by asking an LLM "rate this 0–1." LLM ratings
   aren't reproducible and would drag us into human-agreement studies to defend them.
   We are explicitly **not** claiming these coordinates are human-meaningful scores.
   They're a stable space we can measure and compare in.

2. **An audit for rewrites.** When the rewriter is told "make this more concrete,"
   we want to check — without letting the LLM grade its own work — whether it
   actually moved concreteness, whether it **overshot**, and whether it **drifted**
   onto dimensions it shouldn't have touched. We do this by fingerprinting the
   original and the rewrite and comparing them.

The fingerprint comes from a **sparse autoencoder (SAE)**: a small unsupervised
network that compresses a large neural vector into a sparse set of coordinates. A
useful side effect: average the fingerprints of everything a user engaged with and
you get a **style profile** of that user — concentrated on *how* they like things
written, not *what* they read about.

The real deliverable is not "an SAE runs." It's **a measured answer to whether this
fingerprint actually tracks the nine axes**, in which representation and configuration,
well enough to audit rewrites. The validation is the point.

---

## 2. Two fingerprints, and why we build both

There are two honest ways to fingerprint, and they suit the two jobs differently.
We build and compare both rather than guessing.

- **Single-post fingerprint.** Train the SAE on individual post vectors; fingerprint
  a post by encoding it. The user profile is the mean fingerprint over liked posts.
  This is the natural fit for the **user model** — a liked post stands on its own,
  there's nothing to compare it against.

- **Difference fingerprint.** Train the SAE on the *difference* between two posts;
  fingerprint a change by encoding `vector(rewrite) − vector(original)`. This is the
  natural fit for the **audit** — it reads out exactly which style features the edit
  moved, and it's the cleanest "did it overshoot / drift" signal we can get.

One non-obvious constraint ties these together and is easy to get wrong:

> **An SAE only understands the kind of vector it was trained on.** The encoder is
> `ReLU(Wx + b)`, which is nonlinear, so `f(A − B) ≠ f(A) − f(B)`. If you train on
> single posts you must feed single posts at inference; if you train on differences
> you must feed differences. Mixing them silently produces garbage. We enforce this
> in code with an object-type tag on every trained model (see §7, `validate.py`).

A residual — `post − (mean of its neighbors)` — is itself a difference, which is why
it bridges the two jobs cleanly: a single post becomes a difference-from-its-neighborhood,
so the same manifold serves both the user model and the audit.

---

## 3. Where the style signal lives: space and layer

**Why not just use the BGE embeddings we already have?** BGE-M3 is a *retrieval*
embedding — trained to pull same-topic texts together, which actively squeezes style
into whatever variance is left over. It works as a cheap baseline, but it's the wrong
tool if style is the target. LLM hidden activations are the opposite: a model
predicting the next token has to represent tone, register, stance, and figuration,
so style survives richly. We compare both spaces honestly; expect Qwen to win, but
measure it.

**Why sweep several Qwen layers instead of picking one.** Style isn't parked in a
single known layer, and we shouldn't pretend to know which one. The cleanest evidence
we have (Konen et al., *Style Vectors for Steering Generative LLMs*) finds
style-relevant information robustly encoded across a broad band of the network —
present from early layers, strongest in the **upper-middle**, peaking around layers
18–20 of a ~32-layer model. Qwen2.5-7B has 28 layers, so the corresponding band is
roughly the mid-teens through low-twenties. We extract a few candidate layers
(default 14 / 18 / 22) and let the data choose: the layer whose features best track
the nine axes wins. This turns the layer choice from a guess into a measured
decision, and the extra cost is near-zero because **all candidate layers come out of
a single forward pass** — we hook them simultaneously, not in three separate runs.

**Topic removal: fluid, not schema'd.** To isolate style we subtract the shared
topical component. Two ways, both compared:

- **kNN residual** — subtract the mean of each post's *k* nearest neighbors in the
  representation space. This needs no topic labels at all, which is the point: it
  treats topic as a fluid local thing rather than forcing every post into one of 19
  rigid buckets, and it keeps topic fully decoupled from the code.
- **Raw (no subtraction)** — feed whole vectors and let the SAE separate topic and
  style into different features, then keep the style ones. Topic dominates the
  variance here, so style features tend to be weaker; this is the baseline that tells
  us how much the subtraction actually buys us.

---

## 4. The corpus and the labels

The working corpus is `phase3_curation/out/curated_corpus.parquet`: **9,500 posts,
19 topics, 500 each**, substance-scored 0/1/2, deliberately **not** style-labeled
(style annotation was deferred until the axes settled — they now have). Each row
carries a `row_idx` into the top-level `embeddings.npy`, so the BGE vector for every
post is a slice away — no re-embedding. (Confirmed: `post_ids.npy[row_idx]` equals
`post_id` for all 9,500.)

The SAE trains unsupervised, but to *judge* it we need a reference for "does this
feature track that axis." We use two, because they cover different failure modes:

- **LLM axis scores on a stratified subset.** Continuous scores let us correlate a
  feature's activation against an axis across many posts. We annotate a subset large
  enough to estimate per-axis correlations stably across topics (a couple thousand
  posts, stratified over the 19 topics and substance levels) rather than the whole
  corpus — these are a *reference we don't fully trust*, not training data, so paying
  for all 9,500 buys little. One call per post returns all nine axes; cache the
  results the way phase-3 caches its gate scores so we never re-pay.

- **Controlled synthetic pairs.** Posts deliberately varied along one axis (see §6).
  These give a clean per-axis separation test that doesn't depend on trusting any
  continuous score.

Note the standing question baked into `axes.py`: **`humor` is flagged low-measurability**
— it may be partly absorbed into topic rather than being pure style. Whether the
fingerprint can separate humor at all is one of the concrete things this study
answers; don't assume it works.

---

## 5. Natural vs. synthetic pairs (read this before sizing anything)

Pairs play two completely different roles, and conflating them leads to either wasted
money or a starved model.

- **Training the difference-SAE wants volume.** A dictionary over a 3584-dim space
  needs many thousands of difference vectors to be stable — a few hundred won't do.
  The right source is **natural neighbor pairs**: take embedding-close posts from the
  corpus and difference them. These are essentially free and effectively unlimited —
  from 9,500 posts and their neighborhoods we can draw tens of thousands of
  within-neighborhood differences. This is also conceptually the same "any two close
  posts" idea that motivated the fluid-topic residual.

- **Naming and grading features wants control, not volume.** To say "feature 12
  tracks concreteness" and to measure overshoot/drift, we need pairs that move *one
  named axis* and hold the rest still. Those have to be **LLM-generated**: take a base
  post, ask the rewriter to shift exactly one axis using that axis's definition and
  anchors from `axes.py`, keep topic and length fixed. We generate enough per axis for
  a solid separation estimate (a few hundred clean pairs each), and we spend the calls
  to make them *good* — every pair gets a quick independent check that it moved the
  intended axis and didn't smear others; pairs that fail are dropped. Quality here
  matters more than raw count.

So: **train on natural differences, probe with synthetic single-axis pairs.** If a
particular axis comes out under-recovered, the remedy is to enrich *training* with
synthetic examples for that axis specifically — but that's a targeted fix, not the
default, because we don't want the dictionary memorizing the rewriter's tics.

---

## 6. Build plan

The modules live under `seeding_app/backend/sae/` (alongside the existing residual/
model/train/interpret/correlate code, which we reuse). The earlier 2,550-post
experiments — `m4_sae.py`, `m4b_sae_llm.py` — stay untouched as references. New
config lives in a `SAE2_*` block in `config/settings.py`. Outputs go under
`data/sae2/`.

**Foundations**
- `dataset.py` — assemble the aligned working table from the curated corpus, slice
  the BGE matrix via `row_idx`, and verify alignment against `post_ids.npy`. Everything
  downstream reads its row order from here. *(seeded, working — run it first.)*
- `activations.py` (extend) — `extract_multilayer_activations(...)`: hook the candidate
  Qwen layers at once and mean-pool over real tokens, one forward pass for all layers.
  *(seeded.)*
- `representations.py` — build any input matrix from two levers: space (`bge` |
  `qwen@layer`) × topic-removal (`knn` | `raw`). The kNN residual should visibly shrink
  vector norms relative to raw — that's the cheap check that the subtraction did
  something. *(seeded.)*

**Choosing the layer**
- `layer_sweep.py` — for each candidate Qwen layer, score how well its features track
  the nine axes against the §4 labels (best per-axis `max(|Pearson r|, |axis lift|)`,
  averaged over axes). Pick the winner and carry only it forward. Cite the reason in
  the output so the choice is legible later.

**The fingerprints**
- `run_variant.py` — train one SAE on one representation (reuse `train.py`/`model.py`;
  rescale inputs to unit median norm so the L1 penalty is comparable across spaces).
  Persist weights plus a `meta.json` recording the object type (`single_post` vs
  `pair_difference`), space, layer, topic-removal, and hyperparameters. After the
  layer is chosen, the single-post matrix is `{bge, qwen@best} × {knn, raw}` = four
  variants.

**Knowing whether it works**
- `validate.py` — three things: (a) **coverage** — does every axis have at least one
  feature that tracks it, judged against the LLM-score reference, with the
  `measurability` field from `axes.py` setting expectations (the high-measurability
  axes should come out cleanly; `humor` is the bellwether); (b) **pair separation** —
  the feature that best tracks an axis should separate that axis's synthetic pairs
  (activation gap + ROC-AUC); (c) the **object-type guard** that refuses to fingerprint
  with the wrong kind of input. Plus a small sanity test file (shapes, no NaNs,
  determinism under a fixed seed, norm-shrink for kNN).

**The two uses**
- `usermodel.py` — build a profile as the mean fingerprint over liked posts. Sanity-
  check it by constructing a synthetic user who only liked, say, high-casualness posts
  and confirming the casualness-tracking feature is elevated versus a random user;
  repeat per axis for a recoverability matrix, and bootstrap for stability.
- `pairs.py` — the natural-difference sampler (free, for training the difference-SAE)
  and the synthetic single-axis generator with its quality checks (for probing).
- `audit.py` — given an original, a rewrite, a user profile, and the instructed axes:
  report intended movement (with under/overshoot), drift onto other axes, and whether
  the rewrite moved toward the user's profile. Single-post and difference variants both
  supported, each fed the object type it was trained on. Sanity-check against held-out
  synthetic pairs: a clean rewrite should read as high intended / low drift, a
  deliberately sloppy one should trip the drift flag.

**Tying off**
- `m5_style_audit.py` — orchestrator that trains the variants, runs validation, and
  writes one comparison report: per variant, how many of the nine axes are covered,
  the per-axis numbers, the pair-separation AUCs, and the humor verdict. Sorted by
  coverage. This table is the headline result.
- `api.py` — a thin interface (`get_user_model`, `audit`) so the live app
  (`session.py` / `transform.py`) can call this later. We are **not** wiring it into the
  serving loop yet; this just keeps the seam clean. (When someone does integrate, note
  the stale hardcoded axis list around `evaluation.py:52` — it should read
  `ALL_AXIS_NAMES`.)

Suggested order: `dataset → activations(multilayer) → representations → label subset →
layer_sweep → run_variant → validate (+tests) → pairs → usermodel → audit →
m5_style_audit → api`.

---

## 7. What success looks like

- The high-measurability axes (`casualness`, `reading_level`, `hedging`, `concreteness`)
  are each tracked by at least one feature, in at least one representation.
- A clear, reasoned pick of space + Qwen layer + topic-removal, backed by the
  comparison table rather than taste.
- A defensible verdict on `humor` — recovered or not, with the evidence.
- An audit that, on held-out rewrites, reliably flags overshoot and cross-axis drift.

If some axes don't come out cleanly, that's a real finding, not a failure — it tells
us which delivery dimensions live in the representation and which don't.

---

## 8. Cost and caching

The only steps that cost anything are the subset annotation (§4) and the synthetic
pair generation and checks (§5). Everything else — activation extraction, residuals,
SAE training, all the validation math — is local. Cache every LLM result to disk
(follow the `gate_log.jsonl` pattern in phase-3: keyed, append-only, never re-paid),
so reruns are free and we can grow the labeled/probe sets incrementally. Spend what's
needed to make the labels and pairs *good* — quality of the reference set is what the
whole validation rests on.

---

## 9. Still open (for Rayhan)

- Confirm the nine axes in `axes.py` are the target set as-is (humor stays in, flagged).
- Size of the labeled subset and the per-axis synthetic pair set — defaults are in the
  `SAE2_*` config; bump them if the per-axis estimates look noisy.
- Candidate Qwen layers (default 14 / 18 / 22 of 28) and whether to enrich training
  with synthetic pairs for any axis that comes out weak.
