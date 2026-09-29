"""Compare one video's response profile against a reference, and against history.

Two comparisons, deliberately different in kind:

  * 1:1 — your cut against one reference you are trying to emulate or beat.
    Curves are resampled to a common length so videos of different durations can
    be laid over each other. Position matters more than duration here.

  * history — your cut against your own past evaluations. This is where the
    tool becomes more useful the longer you use it, because past evaluations
    can carry real published performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .features import Features, resample


@dataclass
class SectionDelta:
    label: str
    subject: float
    reference: float
    delta: float
    verdict: str  # ahead | behind | level


@dataclass
class Comparison:
    reference_label: str
    correlation: float          # shape similarity, -1..1
    same_shape: bool
    subject_summary: dict[str, float]
    reference_summary: dict[str, float]
    deltas: dict[str, float]    # headline metric differences
    sections: list[SectionDelta] = field(default_factory=list)
    aligned_subject: list[float] = field(default_factory=list)
    aligned_reference: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sections"] = [
            asdict(s) if not isinstance(s, dict) else s for s in self.sections
        ]
        return d


_HEADLINE = ("opening_2s", "closing_2s", "peak_value", "trough_value", "swing",
             "variability", "decay", "sustained_above")


def _summary(f: Features) -> dict[str, float]:
    return {k: float(getattr(f, k)) for k in _HEADLINE}


def compare(subject: Features, reference: Features, ref_label: str,
            same_shape_r: float = 0.85, points: int = 60) -> Comparison:
    """Lay two response curves over each other and report the differences."""
    a = resample(subject.trace, points)
    b = resample(reference.trace, points)

    r = 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

    sub_s, ref_s = _summary(subject), _summary(reference)
    deltas = {k: round(sub_s[k] - ref_s[k], 4) for k in _HEADLINE}

    # Compare in fifths of each video, so structure lines up proportionally even
    # when the two cuts are different lengths.
    sections: list[SectionDelta] = []
    names = ["first fifth", "second fifth", "middle fifth", "fourth fifth", "final fifth"]
    chunk = points // 5
    for i, name in enumerate(names):
        lo, hi = i * chunk, (i + 1) * chunk if i < 4 else points
        sa, sb = float(a[lo:hi].mean()), float(b[lo:hi].mean())
        d = sa - sb
        verdict = "ahead" if d > 0.05 else ("behind" if d < -0.05 else "level")
        sections.append(
            SectionDelta(name, round(sa, 3), round(sb, 3), round(d, 3), verdict)
        )

    notes: list[str] = []
    if subject.tail_spike or reference.tail_spike:
        notes.append(
            "One or both videos spike on the final step. That is almost always a "
            "hard cut (to black or to a card), not the ending landing — treat it "
            "as an artefact."
        )
    if subject.modalities and reference.modalities and \
            set(subject.modalities) != set(reference.modalities):
        notes.append(
            f"Modality mismatch: subject scored on {', '.join(subject.modalities)}; "
            f"reference on {', '.join(reference.modalities)}. Compare shapes, not "
            "absolute magnitudes."
        )
    if abs(subject.duration_s - reference.duration_s) > max(subject.duration_s, 1) * 0.4:
        notes.append(
            f"Durations differ a lot ({subject.duration_s:.1f}s vs "
            f"{reference.duration_s:.1f}s). Curves were stretched to align; "
            "position is comparable, pacing is not."
        )

    return Comparison(
        reference_label=ref_label,
        correlation=round(r, 3),
        same_shape=bool(r >= same_shape_r),
        subject_summary={k: round(v, 4) for k, v in sub_s.items()},
        reference_summary={k: round(v, 4) for k, v in ref_s.items()},
        deltas=deltas,
        sections=sections,
        aligned_subject=[round(float(x), 4) for x in a],
        aligned_reference=[round(float(x), 4) for x in b],
        notes=notes,
    )


@dataclass
class HistoryPosition:
    """Where this evaluation sits among the user's own past work."""

    n_previous: int
    percentiles: dict[str, float]      # metric -> 0..100 within own history
    best_performing: dict[str, Any] | None   # features of highest-engagement past video
    trend: dict[str, str]              # metric -> improving | declining | flat
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def position_in_history(subject: Features, history: list[dict[str, Any]]) -> HistoryPosition:
    """Rank this video's profile against previous evaluations.

    `history` entries are {"features": Features-dict, "engagement": float|None, ...}
    """
    notes: list[str] = []
    if not history:
        return HistoryPosition(0, {}, None, {}, ["No previous evaluations to compare against yet."])

    pct: dict[str, float] = {}
    for key in _HEADLINE:
        past = [float(h["features"][key]) for h in history if key in h.get("features", {})]
        if not past:
            continue
        cur = float(getattr(subject, key))
        pct[key] = round(100.0 * sum(p < cur for p in past) / len(past), 1)

    # Trend across the last few, oldest -> newest.
    trend: dict[str, str] = {}
    if len(history) >= 3:
        for key in ("opening_2s", "sustained_above", "decay"):
            series = [float(h["features"][key]) for h in history if key in h.get("features", {})]
            if len(series) >= 3:
                half = len(series) // 2
                first = np.mean(series[:half])
                last = np.mean(series[half:])
                diff = last - first
                if key == "decay":  # lower decay is better
                    trend[key] = (
                        "improving" if diff < -0.02
                        else ("declining" if diff > 0.02 else "flat")
                    )
                else:
                    trend[key] = (
                        "improving" if diff > 0.02
                        else ("declining" if diff < -0.02 else "flat")
                    )

    scored = [h for h in history if h.get("engagement") is not None]
    best = None
    if scored:
        best_entry = max(scored, key=lambda h: h["engagement"])
        best = {
            "label": best_entry.get("label"),
            "engagement": best_entry["engagement"],
            "features": {k: best_entry["features"].get(k) for k in _HEADLINE},
        }
        notes.append(
            f"Your best-performing measured video is '{best_entry.get('label') or 'unlabelled'}'."
        )
    else:
        notes.append(
            "No published performance recorded yet. Add views/likes with "
            "`vpa metrics add` and recommendations start learning from outcomes."
        )

    return HistoryPosition(len(history), pct, best, trend, notes)
