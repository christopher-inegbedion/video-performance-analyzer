"""Pull the actual frames at the moments the curve cares about.

A timestamp is not a decision. "Your trough is at 14.4s" makes someone go and
scrub through their own edit to find out what is there; showing the frame closes
that gap, and the report stops being a number and starts being a note about a
shot.

On timing: TRIBE's predictions are already offset to compensate for the
haemodynamic lag, so a curve value at time T corresponds to the stimulus at
time T. No extra correction is applied here — if that ever changes upstream,
this is the module that has to change with it.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analysis.features import Features


@dataclass
class KeyFrame:
    """One extracted frame, with the reason it was worth extracting."""

    label: str
    time_s: float
    value: float          # normalised response at that moment (1.0 = average)
    path: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value_at(features: Features, time_s: float) -> float:
    """Look up the curve value at a timestamp.

    Must ROUND, not truncate. Moment times are produced as `index * step` and
    then rounded to 2dp for display, which loses just enough precision that
    int() lands an index early: a trough at index 5 with step 0.8667 becomes
    4.33s, and int(4.33 / 0.8667) is 4. The report then showed the neighbouring
    value beside a note quoting the real one.
    """
    if not features.trace or features.step_s <= 0:
        return 1.0
    idx = round(time_s / features.step_s)
    idx = max(0, min(idx, len(features.trace) - 1))
    return float(features.trace[idx])


def key_moments(features: Features, include_segments: bool = True) -> list[tuple[str, float, str]]:
    """Decide which moments are worth looking at. Returns (label, time, note)."""
    moments: list[tuple[str, float, str]] = []
    dur = features.duration_s

    moments.append((
        "opening",
        min(0.5, dur / 2),
        "The scroll decision happens here, before anything else matters.",
    ))
    moments.append((
        "peak",
        features.peak_time_s,
        f"Highest predicted response ({features.peak_value:.3f}). Whatever is "
        "working, it is working here.",
    ))
    moments.append((
        "trough",
        features.trough_time_s,
        f"Lowest predicted response ({features.trough_value:.3f}). Usually the "
        "most actionable moment in the video — this is where the edit sags.",
    ))

    if include_segments:
        for seg in features.segments:
            mid = (seg.start_s + seg.end_s) / 2
            moments.append((
                f"section: {seg.label}",
                mid,
                f"Middle of '{seg.label}' ({seg.mean:.3f}, {seg.verdict}).",
            ))

    if dur > 2:
        closing = max(0.0, dur - 1.0)
        note = "Final second."
        if features.tail_spike:
            note += (" The curve spikes here, which is almost certainly a hard "
                     "cut rather than the ending landing.")
        moments.append(("closing", closing, note))

    # Deduplicate moments that land on the same frame — no point extracting the
    # same image three times because the peak happened to be in the opening.
    seen: list[tuple[str, float, str]] = []
    for label, raw_t, note in moments:
        clamped = max(0.0, min(raw_t, max(dur - 0.05, 0.0)))
        if any(abs(clamped - prev_t) < 0.25 for _, prev_t, _ in seen):
            continue
        seen.append((label, clamped, note))
    return seen


def _run(cmd: list[str]) -> bool:
    """Run ffmpeg, reporting success as a bool.

    ffmpeg is an optional runtime dependency: frames are a convenience layered
    on top of an evaluation, and an evaluation is still valid without them. A
    missing binary raises FileNotFoundError from subprocess rather than
    returning non-zero, so it has to be caught explicitly or it takes down the
    whole run on any machine without ffmpeg installed.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    except (FileNotFoundError, OSError):
        return False


def extract(
    video: Path,
    features: Features,
    out_dir: Path,
    include_segments: bool = True,
    width: int = 480,
) -> list[KeyFrame]:
    """Write one JPEG per key moment. Missing frames are skipped, not fatal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[KeyFrame] = []

    for label, time_s, note in key_moments(features, include_segments):
        safe = label.replace(": ", "-").replace(" ", "-").replace("/", "-")
        target = out_dir / f"{safe}_{time_s:06.2f}s.jpg"
        ok = _run([
            "ffmpeg", "-v", "error",
            "-ss", f"{time_s:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            str(target), "-y",
        ])
        if not ok or not target.exists():
            continue
        frames.append(
            KeyFrame(label, round(time_s, 2), round(_value_at(features, time_s), 3),
                     str(target), note)
        )
    return frames


def contact_sheet(frames: list[KeyFrame], target: Path, per_row: int = 4) -> Path | None:
    """Tile the key frames into one image, so the whole shape is visible at once."""
    if not frames:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error"]
    for frame in frames:
        cmd += ["-i", frame.path]

    rows = (len(frames) + per_row - 1) // per_row
    scaled = "".join(f"[{i}]scale=320:-2,pad=330:ih+10:5:5:black[v{i}];"
                     for i in range(len(frames)))
    inputs = "".join(f"[v{i}]" for i in range(len(frames)))
    # xstack needs an exact grid; tile via a filter chain instead so a partial
    # final row is handled without padding the input list with blanks.
    layout = f"{scaled}{inputs}xstack=inputs={len(frames)}:layout="
    positions = []
    for i in range(len(frames)):
        col, row = i % per_row, i // per_row
        positions.append(f"{'0' if col == 0 else '+'.join(['w0'] * col)}_"
                         f"{'0' if row == 0 else '+'.join(['h0'] * row)}")
    layout += "|".join(positions)
    # xstack sizes its canvas to fit every position and fills whatever is left
    # over with green. A partial final row is normal here, so make the gap black.
    layout += ":fill=black"

    cmd += ["-filter_complex", layout, "-frames:v", "1", str(target), "-y"]
    if not _run(cmd) or not target.exists():
        # A contact sheet is a convenience; individual frames are the substance.
        return None
    _ = rows
    return target
