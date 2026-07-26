"""
Bradley-Terry preference model: learns per-axis preference weights from
pairwise/shelf choices via full-covariance multivariate Bayesian logistic
regression (Newton-Raphson MAP fit + Laplace approximation to the posterior).

Parallel to backend/session.py's existing LLM-based update_profile() -- does
NOT replace or modify it, nothing in the live app calls this yet. See
scripts/validate_bt_profile.py for a synthetic-user correctness check.

Model: P(user picks post A over post B) = sigmoid(w . (x_A - x_B)), where x is
a post's 9-axis score vector and w is the user's learned per-axis preference
weight (positive = prefers more of that axis, negative = prefers less).

Full (not diagonal) covariance is required here: the 9 axes are genuinely
correlated in the corpus (e.g. humor/casualness r=0.60) -- an earlier,
simpler version of this model that updated each axis's posterior
independently could not tell "the user actually wants more humor" apart from
"casualness happens to come along with humor in this corpus," and gave large
spurious weights to axes the simulated user was truly indifferent to. Full
covariance lets evidence for one axis correctly discount a merely-correlated
one once the joint fit is refit on all accumulated observations.
"""
import math
import numpy as np
from dataclasses import dataclass, field

AXIS_NAMES = [
    "reading_level", "concreteness", "narrativity", "hedging", "tone",
    "warmth", "self_disclosure", "casualness", "humor",
]
N_AXES = len(AXIS_NAMES)

PRIOR_VARIANCE = 4.0  # w ~ N(0, PRIOR_VARIANCE * I) before any observations
MAX_NEWTON_ITERS = 25
NEWTON_TOL = 1e-6


@dataclass
class BTProfile:
    observations: list = field(default_factory=list)  # list of 9-dim (chosen - rejected) diff arrays
    mu: dict = field(default_factory=lambda: {ax: 0.0 for ax in AXIS_NAMES})
    sigma2: dict = field(default_factory=lambda: {ax: PRIOR_VARIANCE for ax in AXIS_NAMES})
    n_updates: int = 0
    cov: np.ndarray = field(default_factory=lambda: np.eye(N_AXES) * PRIOR_VARIANCE)  # full posterior covariance

    def confidence(self, axis: str) -> float:
        """0-1 confidence, real (not LLM-guessed): rises as the posterior
        marginal variance shrinks from the prior."""
        return float(max(0.0, 1.0 - self.sigma2[axis] / PRIOR_VARIANCE))

    def preferred_direction(self, axis: str) -> str:
        return "increase" if self.mu[axis] >= 0 else "decrease"

    def to_dict(self) -> dict:
        return {"mu": dict(self.mu), "sigma2": dict(self.sigma2), "n_updates": self.n_updates}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _axis_val(axes: dict, ax: str) -> float:
    v = axes.get(ax, 0.5)
    return float(v["score"] if isinstance(v, dict) else v)


def _fit_map(observations: list[np.ndarray], prior_var: float = PRIOR_VARIANCE):
    """
    Newton-Raphson MAP fit of a multivariate logistic regression with a
    Gaussian prior N(0, prior_var*I). Every stored observation is a
    (chosen - rejected) diff vector with an implicit outcome=1 (the model
    always "explains" why the chosen post won), so P(outcome=1 | w) =
    sigmoid(w . d). Returns (posterior mean w, posterior covariance).
    """
    X = np.array(observations, dtype=np.float64)  # (n_obs, N_AXES)
    w = np.zeros(N_AXES)
    prior_precision = np.eye(N_AXES) / prior_var

    for _ in range(MAX_NEWTON_ITERS):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (1.0 - p) - w / prior_var           # gradient of log posterior
        neg_hessian = (X.T * (p * (1.0 - p))) @ X + prior_precision  # -Hessian, positive definite
        delta = np.linalg.solve(neg_hessian, grad)
        w = w + delta
        if np.linalg.norm(delta) < NEWTON_TOL:
            break

    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    neg_hessian = (X.T * (p * (1.0 - p))) @ X + prior_precision
    cov = np.linalg.inv(neg_hessian)  # Laplace approximation to the posterior covariance
    return w, cov


def update_pairwise(profile: BTProfile, chosen_axes: dict, rejected_axes: dict) -> BTProfile:
    """One observed pairwise comparison: refits the full joint model on every
    observation seen so far (cheap at session scale -- a handful of ms for
    hundreds of 9-dim observations)."""
    d = np.array([_axis_val(chosen_axes, ax) - _axis_val(rejected_axes, ax) for ax in AXIS_NAMES])
    profile.observations.append(d)
    profile.n_updates += 1

    if not np.any(d):
        return profile  # this pair carries no information on any axis

    w, cov = _fit_map(profile.observations)
    profile.cov = cov
    for i, ax in enumerate(AXIS_NAMES):
        profile.mu[ax] = float(w[i])
        profile.sigma2[ax] = float(cov[i, i])
    return profile


def update_shelf(profile: BTProfile, shown_posts: list[dict], chosen_ids: list[str]) -> BTProfile:
    """Shelf mode: expand into one pairwise "chosen preferred over skipped"
    observation per (chosen, skipped) combination from the same shown set."""
    chosen = [p for p in shown_posts if p["post_id"] in chosen_ids]
    skipped = [p for p in shown_posts if p["post_id"] not in chosen_ids]
    for c in chosen:
        for s in skipped:
            update_pairwise(profile, c.get("axes", {}), s.get("axes", {}))
    return profile


def select_next_pair(profile: BTProfile, candidate_posts: list[dict],
                      max_candidates: int = 40, rng=None):
    """
    D-optimal / max-information pair selection (Mukherjee et al. 2024,
    "Optimal Design for Human Preference Elicitation", arXiv 2404.13895):
    choose the candidate pair whose axis-score difference best reduces the
    model's CURRENT posterior uncertainty, instead of a random or merely-
    similar pair.

    Why this exists: correlated-but-passively-sampled pairs (e.g. two posts
    that happen to be similar on every axis except the one you care about,
    drawn from the natural corpus) don't carry enough independent
    information to disentangle genuinely correlated axes (humor/casualness,
    r=0.60 in this corpus) -- validated empirically, see
    scripts/validate_bt_profile.py and sae_verification_findings memory.
    Fixing that requires choosing pairs deliberately, not fitting harder on
    whatever pairs were shown.

    Score for a candidate pair (i, j): d^T . Sigma . d, where d is the axis-
    score difference and Sigma is profile.cov (the model's current posterior
    covariance). This is the classical D-optimal-design criterion in
    quadratic form: larger score = this pair varies most along the
    direction the model is currently LEAST certain about, so observing it
    teaches the model the most (equivalent to maximizing det(V + d d^T) via
    the matrix determinant lemma).
    """
    import random
    rng = rng or random.Random()

    pool = candidate_posts
    if len(pool) > max_candidates:
        pool = rng.sample(pool, max_candidates)

    best_score = -1.0
    best_pair = None
    for a_idx in range(len(pool)):
        for b_idx in range(a_idx + 1, len(pool)):
            post_a, post_b = pool[a_idx], pool[b_idx]
            d = np.array([_axis_val(post_a.get("axes", {}), ax) - _axis_val(post_b.get("axes", {}), ax)
                          for ax in AXIS_NAMES])
            if not np.any(d):
                continue
            score = float(d @ profile.cov @ d)
            if score > best_score:
                best_score = score
                best_pair = (post_a, post_b)

    return best_pair


def preferred_value(profile: BTProfile, axis: str) -> float:
    """
    Squash a learned preference weight into a 0-1 "preferred value" comparable
    to a post's own axis score (which is what the existing rewrite-target
    logic in transform.py's compute_transform_deltas() expects) -- mu=0 (no
    preference) -> 0.5 (neutral), mu>0 -> closer to 1, mu<0 -> closer to 0.
    """
    return 0.5 + 0.5 * math.tanh(profile.mu[axis] / 2.0)


def compute_all_preferred_values(profile: BTProfile) -> dict:
    """
    Raw preferred_value() for EVERY axis, completely unfiltered -- no
    confidence floor, no top-N ranking, no neutral-band exclusion. This is
    NOT what drives rewriting (that's compute_bt_prefs(), which filters down
    to a handful of confident + decisive axes) -- it exists purely so a dev/
    debug view can show the model's full current belief on every axis, not
    just the subset currently deemed good enough to act on.
    """
    return {ax: round(preferred_value(profile, ax), 4) for ax in AXIS_NAMES}


def to_deltas(profile: BTProfile, target_axes: list[str] | None = None,
              min_confidence: float = 0.0) -> dict:
    """
    Learned preference weights -> the same shape transform_user()'s deltas
    dict expects, minus current/target (the caller fills those in per
    specific post being rewritten). Only includes axes whose confidence
    clears min_confidence.
    """
    axes_to_use = target_axes or AXIS_NAMES
    out = {}
    for ax in axes_to_use:
        conf = profile.confidence(ax)
        if conf < min_confidence:
            continue
        out[ax] = {"direction": profile.preferred_direction(ax), "weight": profile.mu[ax], "confidence": conf}
    return out


# Confidence floor before trusting an axis enough to act on it. Empirically,
# threshold-gating alone was found NOT to reliably separate correct from
# confidently-wrong axes under realistic noise (see bt_preference_model
# memory) -- this floor is a coarse, honest-about-its-limits filter, not a
# guarantee, and n_updates is checked alongside confidence for the same reason.
MIN_CONFIDENCE_FOR_PREFS = 0.3
MIN_UPDATES_FOR_PREFS = 5

# Cap on how many axes get selected as rewrite targets. Rewrites only ever
# target 1-2 axes at a time anyway (compute_transform_deltas's max_axes), so
# there's no benefit to feeding it a style_prefs dict wider than a handful of
# axes -- narrowing the candidate pool to the most confident ones keeps
# rewrite targeting focused on the preferences the model is most sure about.
MAX_REWRITE_AXES = 4


def compute_bt_prefs(profile: BTProfile, min_confidence: float = MIN_CONFIDENCE_FOR_PREFS,
                      min_updates: int = MIN_UPDATES_FOR_PREFS, max_axes: int = MAX_REWRITE_AXES) -> dict:
    """
    Learned preference weights -> {axis: preferred_value in [0,1]}, for axes
    confident enough to act on. This is what replaces the old SAE/Ridge-
    fingerprint-based compute_sae_prefs() as the source of rewrite-target
    values fed into transform.py.

    Selection: rank every axis that clears min_confidence by confidence,
    descending, and keep the top max_axes.
    """
    if profile.n_updates < min_updates:
        return {}

    ranked = sorted(
        (ax for ax in AXIS_NAMES if profile.confidence(ax) >= min_confidence),
        key=lambda ax: -profile.confidence(ax),
    )[:max_axes]

    return {ax: round(preferred_value(profile, ax), 4) for ax in ranked}
