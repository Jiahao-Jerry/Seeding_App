"""
Admin settings for the evaluation/study runtime.
These control behavior without code changes — toggled via the admin dashboard.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class AdminSettings:
    # ── Trial construction ────────────────────────────────────────
    n_type_a: int = 8           # same-band trials
    n_type_b: int = 8           # reach trials (transformed-mid vs original-near)
    n_type_c: int = 0           # same-post trials (0 = disabled)

    # ── Transformation ────────────────────────────────────────────
    max_axes: int = 2           # max axes to transform per post (1-2 = natural)
    verify_transform: bool = False  # run LLM verification (doubles LLM calls)
    transform_threshold: float = 0.15  # min gap to trigger transform on an axis

    # ── Post selection ────────────────────────────────────────────
    prefer_high_gap: bool = True  # within a band, prefer posts with large style mismatch
    diversify_topics: bool = True  # ensure topic diversity in trial set

    # ── Results display ───────────────────────────────────────────
    show_axis_breakdown: bool = True   # show per-axis stats on results screen
    show_band_breakdown: bool = True   # show per-band stats
    show_feed_button: bool = False     # show "See Feed" button after seeding (demo only)

    # ── Profile learning ──────────────────────────────────────────
    update_profile_from_eval: bool = False  # feed eval choices back into profile

    def to_dict(self):
        return asdict(self)


# Singleton — modified via API
_settings = AdminSettings()


def get_settings() -> AdminSettings:
    return _settings


def update_settings(**kwargs) -> AdminSettings:
    global _settings
    for k, v in kwargs.items():
        if hasattr(_settings, k):
            setattr(_settings, k, v)
    return _settings
