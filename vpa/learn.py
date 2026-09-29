"""The self-improving part: correlate predicted features with real outcomes.

The mechanism is deliberately simple and honest about its own weakness. Every
evaluation stores a feature profile. Every published video can have views/likes
attached. Join those two and you get (predicted shape -> what actually happened)
pairs.

With a handful of pairs that is anecdote, not evidence, and this module says so
rather than dressing up noise as insight. With enough pairs it becomes the most
valuable thing the tool knows, because it is grounded in YOUR audience rather
than a general model of brains.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from . import db

# Below this many labelled videos we refuse to report correlations as findings.
MIN_FOR_SIGNAL = 5
# Below this we won't even hint at direction.
MIN_FOR_HINT = 3

_FEATURES = ("opening_2s", "closing_2s", "peak_value", "trough_value", "swing",
             "variability", "decay", "sustained_above")


def engagement_rate(row: Any) -> float | None:
    """Engagement per view — comparable across videos with different reach.

    Falls back to raw likes when views are missing, and returns None when there
    is nothing usable, so callers can skip rather than invent a number.
    """
    views = row["views"] if isinstance(row, dict) else row["views"]
    likes = row["likes"] if isinstance(row, dict) else row["likes"]
    comments = (row["comments"] if isinstance(row, dict) else row["comments"]) or 0
    shares = (row["shares"] if isinstance(row, dict) else row["shares"]) or 0
    saves = (row["saves"] if isinstance(row, dict) else row["saves"]) or 0
    interactions = (likes or 0) + comments + shares + saves
    if views and views > 0:
        return round(interactions / views, 6)
    if interactions:
        return float(interactions)
    return None


@dataclass
class Correlation:
    feature: str
    r: float
    n: int
    direction: str      # higher-is-better | lower-is-better | unclear
    confidence: str     # none | weak | moderate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LearnedModel:
    n_labelled: int
    correlations: list[Correlation] = field(default_factory=list)
    best: dict[str, Any] | None = None
    worst: dict[str, Any] | None = None
    caveat: str = ""
    usable: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["correlations"] = [c.as_dict() if not isinstance(c, dict) else c
                             for c in self.correlations]
        return d

    def summary_text(self) -> str:
        if not self.n_labelled:
            return ("No published performance recorded yet, so recommendations are based "
                    "on the model's predictions alone.")
        lines = [f"Learned from {self.n_labelled} video(s) with recorded performance."]
        if not self.usable:
            lines.append(self.caveat)
        for c in self.correlations[:5]:
            arrow = "higher" if c.r > 0 else "lower"
            lines.append(
                f"  {c.feature}: r={c.r:+.2f} (n={c.n}, {c.confidence}) — "
                f"{arrow} values went with more engagement"
            )
        if self.best:
            lines.append(f"  Best performer: {self.best.get('label') or 'unlabelled'}")
        return "\n".join(lines)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def build(path=None) -> LearnedModel:
    """Fit the (very small) model relating predicted features to real outcomes."""
    rows = db.labelled_history(path)
    pairs: list[tuple[dict, float, str]] = []
    for r in rows:
        try:
            feats = json.loads(r["features"]) if r["features"] else None
        except json.JSONDecodeError:
            continue
        if not feats:
            continue
        eng = engagement_rate(r)
        if eng is None:
            continue
        pairs.append((feats, eng, r["label"] or r["video_id"]))

    n = len(pairs)
    if n == 0:
        return LearnedModel(0, caveat="No labelled videos yet.", usable=False)

    correlations: list[Correlation] = []
    if n >= MIN_FOR_HINT:
        engs = [p[1] for p in pairs]
        for feat in _FEATURES:
            xs = [float(p[0].get(feat, 0.0)) for p in pairs]
            if len(set(xs)) < 2:
                continue
            r = _pearson(xs, engs)
            confidence = "none"
            if n >= MIN_FOR_SIGNAL and abs(r) >= 0.5:
                confidence = "moderate"
            elif n >= MIN_FOR_HINT and abs(r) >= 0.35:
                confidence = "weak"
            direction = "higher-is-better" if r > 0.15 else (
                "lower-is-better" if r < -0.15 else "unclear")
            correlations.append(
                Correlation(feat, round(r, 3), n, direction, confidence)
            )
        correlations.sort(key=lambda c: abs(c.r), reverse=True)

    ranked = sorted(pairs, key=lambda p: p[1])
    best = {"label": ranked[-1][2], "engagement": ranked[-1][1],
            "features": {k: ranked[-1][0].get(k) for k in _FEATURES}}
    worst = {"label": ranked[0][2], "engagement": ranked[0][1],
             "features": {k: ranked[0][0].get(k) for k in _FEATURES}}

    usable = n >= MIN_FOR_SIGNAL
    caveat = (
        f"Only {n} labelled video(s). With fewer than {MIN_FOR_SIGNAL} these "
        "correlations are anecdote, not evidence — they are shown so you can see "
        "the tool learning, and should not drive decisions yet."
    ) if not usable else (
        f"Based on {n} labelled videos. Still a small sample: treat these as "
        "hypotheses about your audience, not laws."
    )

    return LearnedModel(n, correlations, best, worst, caveat, usable)


def history_for_comparison(path=None) -> list[dict[str, Any]]:
    """Past evaluations in the shape `compare.position_in_history` expects."""
    out: list[dict[str, Any]] = []
    for r in db.labelled_history(path):
        try:
            feats = json.loads(r["features"]) if r["features"] else None
        except json.JSONDecodeError:
            continue
        if feats:
            out.append(
                {
                    "features": feats,
                    "engagement": engagement_rate(r),
                    "label": r["label"],
                    "evaluation_id": r["evaluation_id"],
                }
            )
    # Also include completed evaluations with no metrics — useful for shape
    # comparison even when we can't learn from outcomes.
    seen = {o["evaluation_id"] for o in out}
    for e in db.list_evaluations(limit=200, status="done", path=path):
        if e["id"] in seen or not e["features"]:
            continue
        try:
            feats = json.loads(e["features"])
        except json.JSONDecodeError:
            continue
        out.append(
            {"features": feats, "engagement": None,
             "label": e["video_label"], "evaluation_id": e["id"]}
        )
    return out


def stale_evaluations(path=None) -> list[dict[str, Any]]:
    """Evaluations whose advice predates what we now know.

    An evaluation is stale if new performance data has been recorded since its
    most recent recommendation was written. Those are the ones worth revisiting.
    """
    stale: list[dict[str, Any]] = []
    with db.connect(path) as conn:
        rows = conn.execute(
            """SELECT e.id, e.video_id, v.label,
                      (SELECT MAX(created_at) FROM recommendations r
                        WHERE r.evaluation_id = e.id) AS last_rec,
                      (SELECT MAX(recorded_at) FROM performance p) AS last_metric,
                      (SELECT COUNT(*) FROM recommendations r
                        WHERE r.evaluation_id = e.id) AS n_recs
               FROM evaluations e
               JOIN videos v ON v.id = e.video_id
               WHERE e.status='done'"""
        ).fetchall()
    for r in rows:
        if not r["last_metric"]:
            continue
        if r["n_recs"] == 0:
            continue
        if r["last_rec"] and r["last_metric"] > r["last_rec"]:
            stale.append(
                {"evaluation_id": r["id"], "label": r["label"],
                 "last_rec": r["last_rec"], "last_metric": r["last_metric"]}
            )
    return stale
