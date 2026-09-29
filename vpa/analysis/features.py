"""Reduce a (timesteps, 20484) prediction array to an interpretable profile.

The raw model output is one predicted activation value per cortical vertex per
half-second. Nobody can read that. What survives compression, and what actually
varies between edits, is the *shape* of total predicted response over time.

Everything here is normalised to the video's own mean, so a value of 1.0 means
"average for this video". That matters: absolute magnitudes are not comparable
between videos scored with different modalities or frame rates, but shapes are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Segment:
    """A named stretch of the video, for reporting against structure."""

    label: str
    start_s: float
    end_s: float
    mean: float
    verdict: str  # strong | average | weak


@dataclass
class Features:
    n_steps: int
    duration_s: float
    step_s: float
    mean_magnitude: float
    trace: list[float]              # normalised, one per timestep
    opening_2s: float
    closing_2s: float
    peak_value: float
    peak_time_s: float
    trough_value: float
    trough_time_s: float
    swing: float                    # peak - trough
    variability: float              # std of normalised trace
    decay: float                    # opening - closing; positive means it sheds
    sustained_above: float          # fraction of video above its own mean
    segments: list[Segment] = field(default_factory=list)
    tail_spike: bool = False        # final step jumps (usually a cut to black)
    modalities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["segments"] = [asdict(s) if not isinstance(s, dict) else s for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Features:
        d = dict(d)
        d["segments"] = [Segment(**s) if isinstance(s, dict) else s for s in d.get("segments", [])]
        return cls(**d)


def magnitude_trace(preds: np.ndarray) -> np.ndarray:
    """Total predicted cortical response per timestep (L2 norm across vertices)."""
    return np.linalg.norm(preds, axis=1)


def extract(
    preds: np.ndarray,
    duration_s: float,
    modalities: list[str] | None = None,
    strong: float = 1.10,
    weak: float = 0.90,
    ignore_tail_s: float = 1.5,
    segments: list[tuple[str, float, float]] | None = None,
) -> Features:
    """Build the interpretable profile from raw predictions."""
    mag = magnitude_trace(preds)
    mean_mag = float(mag.mean()) if mag.size else 0.0
    norm = mag / (mean_mag or 1.0)
    n = len(norm)
    step = duration_s / n if n else 0.0

    def window(a: float, b: float) -> float:
        lo, hi = int(a / step) if step else 0, int(b / step) if step else n
        lo = max(0, min(lo, n - 1))
        hi = max(lo + 1, min(hi, n))
        return float(norm[lo:hi].mean())

    # Exclude the tail when hunting the peak. A cut to black produces a hard
    # visual discontinuity and always spikes; reporting that as "your ending
    # works" would be actively misleading.
    keep = max(1, n - int(ignore_tail_s / step)) if step else n
    body = norm[:keep]
    peak_i = int(np.argmax(body))
    trough_i = int(np.argmin(body))

    tail_spike = bool(n >= 3 and norm[-1] > max(1.15, float(body.max()) * 0.98))

    seg_objs: list[Segment] = []
    for label, a, b in segments or []:
        m = window(a, b)
        verdict = "strong" if m >= strong else ("weak" if m <= weak else "average")
        seg_objs.append(Segment(label=label, start_s=a, end_s=b, mean=round(m, 3), verdict=verdict))

    opening = window(0.0, min(2.0, duration_s))
    closing = window(max(0.0, duration_s - 2.0), duration_s)

    return Features(
        n_steps=n,
        duration_s=round(duration_s, 3),
        step_s=round(step, 4),
        mean_magnitude=round(mean_mag, 4),
        trace=[round(float(v), 4) for v in norm],
        opening_2s=round(opening, 4),
        closing_2s=round(closing, 4),
        peak_value=round(float(body[peak_i]), 4),
        peak_time_s=round(peak_i * step, 2),
        trough_value=round(float(body[trough_i]), 4),
        trough_time_s=round(trough_i * step, 2),
        swing=round(float(body.max() - body.min()), 4),
        variability=round(float(norm.std()), 4),
        decay=round(opening - closing, 4),
        sustained_above=round(float((norm > 1.0).mean()), 4),
        tail_spike=tail_spike,
        modalities=modalities or [],
        segments=seg_objs,
    )


def auto_segments(duration_s: float, n: int = 4) -> list[tuple[str, float, float]]:
    """Even thirds/quarters when the user hasn't described their own structure."""
    names = {
        3: ["opening", "middle", "ending"],
        4: ["opening", "early-middle", "late-middle", "ending"],
        5: ["opening", "build", "middle", "turn", "ending"],
    }.get(n, [f"part {i+1}" for i in range(n)])
    edges = np.linspace(0, duration_s, n + 1)
    return [(names[i], float(edges[i]), float(edges[i + 1])) for i in range(n)]


def parse_segments(spec: str, duration_s: float) -> list[tuple[str, float, float]]:
    """Parse `--segments "hook:0-3,montage:3-16,card:16-19,end:19-24"`."""
    out: list[tuple[str, float, float]] = []
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        label, _, span = chunk.partition(":")
        a, _, b = span.partition("-")
        try:
            start = float(a)
            end = float(b) if b else duration_s
        except ValueError as exc:
            raise ValueError(f"Could not read segment '{chunk}' (expected name:start-end)") from exc
        out.append((label.strip() or "segment", start, min(end, duration_s)))
    return out


def resample(trace: list[float] | np.ndarray, n: int) -> np.ndarray:
    """Stretch/squash a trace to n points so two videos can be compared."""
    arr = np.asarray(trace, dtype=float)
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.repeat(arr, n)[:n]
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, arr)
