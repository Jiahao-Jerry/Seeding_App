"""
Delivery axes — the single source of truth. Nothing about an axis is hardcoded
elsewhere. All axes are scalar (0.0–1.0) with optional subtype labels.

These axes are NOT brainstormed; they are DERIVED. The derivation is the
justification — it answers "why these and not others?"

─────────────────────────────────────────────────────────────────────────────
THE GENERATIVE PRINCIPLE
  Hold WHAT is said fixed (topic); vary HOW it is said (delivery). The "how"
  space is partitioned by functional linguistics — every utterance serves three
  functions at once (Halliday):
    - ideational    → the content/topic                  (we factor this OUT)
    - interpersonal → the speaker–listener relationship  (stance, affect, register)
    - textual       → how the message is packaged        (structure, density)
  plus Jakobson's poetic function (focus on the message's form: humor, figuration).
  Biber's empirical register dimensions independently recover the same structure,
  and our own PCA echoes them — evidence the axes are real, not invented.

THE QUESTION SET (the plain-language form of the same decomposition)
  Every post answers a handful of questions about HOW; each axis is one answer:
    textual        How hard to take in?      → reading_level
                   How is it backed up?       → concreteness
                   How is it structured?      → narrativity
    interpersonal  How certain does it sound? → hedging
                   How charged is it?         → tone
                   How warm toward you?       → warmth
                   How much of the author?    → self_disclosure
                   How formal or casual?      → casualness
    poetic         How playful/dressed-up?    → humor

THREE-STAGE JUSTIFICATION (why exactly these survive)
  1. GENERATE candidates from the function taxonomy above, so coverage is
     principled and gaps are VISIBLE (the taxonomy is what flagged the missing
     'hedging'/modality axis).
  2. FILTER by four operational constraints:
       (a) delivery, not content  — survives a topic swap
       (b) transformable          — an LLM can move it without changing facts
       (c) measurable             — shows up as a direction separable from topic
                                     (humor is the known risk — flagged below)
       (d) preference-bearing     — a user could plausibly hold a stable taste for it
  3. VERIFY independence empirically (correlation + PCA on LLM scores). This set
     merged abstract_concrete+grounding (r=0.92), folded background into
     reading_level (r=0.64), and dropped low-variance / low-measurability candidates.

DELIBERATE EXCLUSIONS (documented, not accidental)
  - opinionated / stance (appraisal): high user value but content-entangled and
    fuses with tone (r=0.67) → excluded on measurability; revisit if needed.
  - addressivity / mood (conative): a real interpersonal family, but near-constant
    in this broadcast-style corpus → underweighted.
  - length (textual): a real lever, but this corpus is ~300-char capped so it
    cannot be tested here → excluded as UNTESTABLE, not as unreal.

SCHEMA per axis:
  name          unique key (used everywhere)
  type          "scalar"
  additive      True if INCREASING it tends to import material not in the original
                (higher fidelity risk; consumed by transform.py)
  function      Halliday bucket: "textual" | "interpersonal" | "poetic"
  question      the generating question this axis answers
  measurability "high" | "medium" | "low" — confidence it is separable from topic
  definition    plain definition with low/high anchors
"""

AXES = [
    # ── textual: how the message is packaged ────────────────────────────────
    {
        "name": "reading_level",
        "type": "scalar",
        "additive": False,
        "function": "textual",
        "question": "How hard is it to take in?",
        "measurability": "high",
        "definition": "How advanced the vocabulary and sentence structure is, "
                      "from very simple everyday language (0) to specialist "
                      "jargon and complex syntax (1). Surface form only — folds "
                      "in the old 'background' axis (how much context is assumed).",
    },
    {
        "name": "concreteness",
        "type": "scalar",
        "additive": True,
        "function": "textual",
        "question": "How is it backed up — abstract claim or specific cases?",
        "measurability": "high",
        "definition": "Abstract assertions with no examples (0) vs concrete "
                      "details, specific cases, numbers, named things, analogies "
                      "(1). Merges the old 'abstract_concrete' and 'grounding' "
                      "axes (they correlated r=0.92).",
    },
    {
        "name": "narrativity",
        "type": "scalar",
        "additive": False,
        "function": "textual",
        "question": "How is it structured — argument or story?",
        "measurability": "medium",
        "definition": "Pure assertion/argument (0) vs story-like delivery with "
                      "characters, sequence, scene, or arc (1). Subtype captures "
                      "kind: personal anecdote, news narrative, historical, etc.",
    },
    # ── interpersonal: the speaker–listener relationship ────────────────────
    {
        "name": "hedging",
        "type": "scalar",
        "additive": False,
        "function": "interpersonal",
        "question": "How certain does it sound?",
        "measurability": "high",
        "definition": "States things flatly as fact (0) vs heavily qualified "
                      "with hedges and uncertainty markers — 'might', 'perhaps', "
                      "'I think', 'arguably', 'seems' (1). Epistemic modality; "
                      "independent of whether the claim is actually true.",
    },
    {
        "name": "tone",
        "type": "scalar",
        "additive": False,
        "function": "interpersonal",
        "question": "How emotionally charged is it?",
        "measurability": "medium",
        "definition": "Calm, measured, detached (0) vs emotionally charged, "
                      "urgent, intense (1). This is INTENSITY of feeling only — "
                      "stance/opinion is NOT part of it (a calm post can hold a "
                      "strong opinion).",
    },
    {
        "name": "warmth",
        "type": "scalar",
        "additive": False,
        "function": "interpersonal",
        "question": "How warm is it toward the reader or subject?",
        "measurability": "medium",
        "definition": "Cold, harsh, or hostile (0) vs warm, kind, affirming, "
                      "generous (1). Direction of affect toward the reader/subject "
                      "— distinct from tone, which is intensity regardless of "
                      "direction.",
    },
    {
        "name": "self_disclosure",
        "type": "scalar",
        "additive": True,
        "function": "interpersonal",
        "question": "How much of the author's own self is in it?",
        "measurability": "medium",
        "definition": "Impersonal and detached (0) vs openly personal, "
                      "confessional, sharing the author's own experience or "
                      "feelings (1). Distinct from narrativity — you can confess "
                      "without a story, or narrate someone else without disclosing.",
    },
    {
        "name": "casualness",
        "type": "scalar",
        "additive": False,
        "function": "interpersonal",
        "question": "How formal or casual is the register?",
        "measurability": "high",
        "definition": "Polished, formal written register (0) vs casual internet "
                      "register — slang, abbreviations, lowercase, fragments, "
                      "emoji (1). Pure register/tenor; the most topic-independent, "
                      "easiest-to-measure axis on the list.",
    },
    # ── poetic: focus on the message's form ─────────────────────────────────
    {
        "name": "humor",
        "type": "scalar",
        "additive": True,
        "function": "poetic",
        "question": "How playful or dressed-up is it?",
        "measurability": "low",
        "definition": "Fully serious (0) vs primarily playful/comedic (1). "
                      "Subtype captures kind: dry, satirical, self-deprecating, "
                      "absurdist, deadpan, witty. MEASURABILITY RISK: humor may be "
                      "partly absorbed into topic rather than being pure style "
                      "(flagged by Farnaz) — verify separability before relying on it.",
    },
]

# All axis names (for iteration)
ALL_AXIS_NAMES = [a["name"] for a in AXES]

# Subtractive axes — safe core for transformation (low fidelity risk)
SUBTRACTIVE_AXES = [a["name"] for a in AXES if not a["additive"]]

# Additive axes — increasing them may import material not in the original
# (higher fidelity risk)
ADDITIVE_AXES = [a["name"] for a in AXES if a["additive"]]

# Grouped by Halliday function (for documentation / coverage checks)
AXES_BY_FUNCTION = {
    "textual": [a["name"] for a in AXES if a["function"] == "textual"],
    "interpersonal": [a["name"] for a in AXES if a["function"] == "interpersonal"],
    "poetic": [a["name"] for a in AXES if a["function"] == "poetic"],
}
