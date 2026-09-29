"""Key-frame selection: which moments are worth looking at, and why."""

from __future__ import annotations

import numpy as np
import pytest

from vpa import frames
from vpa.analysis import features as feat


def make(trace, duration=10.0, segments=None):
    preds = np.ones((len(trace), 16)) * np.asarray(trace, dtype=float).reshape(-1, 1)
    return feat.extract(preds, duration_s=duration, segments=segments)


def labels_of(moments):
    return [m[0] for m in moments]


def test_always_offers_opening_peak_and_trough():
    moments = frames.key_moments(make([1, 3, 1, 2, 1]))
    for expected in ("opening", "peak", "trough"):
        assert expected in labels_of(moments)


def test_peak_and_trough_land_on_the_right_times():
    f = make([1, 1, 5, 1, 1, 0.2, 1, 1])  # peak at index 2, trough at index 5
    moments = {m[0]: m[1] for m in frames.key_moments(f)}
    assert moments["peak"] == pytest.approx(f.peak_time_s)
    assert moments["trough"] == pytest.approx(f.trough_time_s)


def test_segments_become_their_own_moments():
    segs = feat.parse_segments("hook:0-3,body:3-7,end:7-10", 10.0)
    f = make([1, 2, 1, 1, 1, 1, 1, 2, 1, 1], segments=segs)
    got = labels_of(frames.key_moments(f))
    assert "section: hook" in got
    assert "section: body" in got


def test_segments_can_be_excluded():
    segs = feat.parse_segments("hook:0-3,body:3-10", 10.0)
    f = make([1, 2, 1, 1, 1, 1, 1, 1, 1, 1], segments=segs)
    got = labels_of(frames.key_moments(f, include_segments=False))
    assert not any(g.startswith("section:") for g in got)


def test_moments_never_fall_outside_the_video():
    f = make([1, 2, 3], duration=3.0)
    for _, time_s, _ in frames.key_moments(f):
        assert 0.0 <= time_s <= 3.0


def test_near_identical_moments_are_deduplicated():
    """Peak inside the opening should not produce two extractions of one frame."""
    f = make([5, 1, 1, 1, 1], duration=5.0)   # peak at 0.0s, opening at 0.5s
    times = [m[1] for m in frames.key_moments(f)]
    assert len(times) == len(set(times))
    for i, a in enumerate(times):
        for b in times[i + 1:]:
            assert abs(a - b) >= 0.25


def test_tail_spike_is_called_out_in_the_note():
    f = make([1.0] * 9 + [8.0], duration=10.0)
    assert f.tail_spike
    closing = next(m for m in frames.key_moments(f) if m[0] == "closing")
    assert "cut" in closing[2].lower()


def test_every_moment_carries_a_reason():
    segs = feat.parse_segments("hook:0-5,end:5-10", 10.0)
    f = make([1, 2, 1, 1, 3, 1, 1, 1, 0.5, 1], segments=segs)
    for label, _, note in frames.key_moments(f):
        assert note.strip(), f"{label} has no explanation"


def test_contact_sheet_of_nothing_is_none(tmp_path):
    assert frames.contact_sheet([], tmp_path / "sheet.jpg") is None


def test_extract_is_not_fatal_when_the_video_is_missing(tmp_path):
    """A broken video should degrade to no frames, not kill the evaluation."""
    f = make([1, 2, 1])
    got = frames.extract(tmp_path / "does-not-exist.mp4", f, tmp_path / "out")
    assert got == []


def test_reported_value_matches_the_feature_it_came_from():
    """The bug this guards: the frame value disagreed with its own note.

    Moment times are `index * step` rounded to 2dp. Truncating on the way back
    lands an index early whenever that rounding loses precision, so the report
    showed one value in the heading and a different one in the explanation.
    """
    trace = [0.93, 1.16, 1.12, 1.04, 0.93, 0.70, 1.14]
    f = make(trace, duration=6.067)

    assert f.trough_value == pytest.approx(min(trace), abs=0.01)
    assert frames._value_at(f, f.trough_time_s) == pytest.approx(f.trough_value, abs=0.01)
    assert frames._value_at(f, f.peak_time_s) == pytest.approx(f.peak_value, abs=0.01)


@pytest.mark.parametrize("n_steps", [3, 5, 7, 11, 13, 25])
def test_value_lookup_round_trips_at_any_resolution(n_steps):
    """Non-dividing step sizes are the normal case, not an edge case."""
    trace = [1.0 + (i % 3) * 0.1 for i in range(n_steps)]
    f = make(trace, duration=6.067)
    assert frames._value_at(f, f.peak_time_s) == pytest.approx(f.peak_value, abs=0.01)
    assert frames._value_at(f, f.trough_time_s) == pytest.approx(f.trough_value, abs=0.01)


def test_extract_degrades_when_ffmpeg_is_not_installed(tmp_path, monkeypatch):
    """ffmpeg is optional. Without it we lose frames, not the whole evaluation.

    subprocess raises FileNotFoundError for a missing binary rather than
    returning non-zero, so this path is distinct from "ffmpeg ran and failed"
    and is invisible on any machine that happens to have ffmpeg installed.
    """
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'ffmpeg'")

    monkeypatch.setattr(frames.subprocess, "run", missing)
    f = make([1, 2, 1])
    assert frames.extract(tmp_path / "clip.mp4", f, tmp_path / "out") == []


def test_contact_sheet_degrades_when_ffmpeg_is_not_installed(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'ffmpeg'")

    monkeypatch.setattr(frames.subprocess, "run", missing)
    shot = frames.KeyFrame("peak", 1.0, 0.9, str(tmp_path / "a.jpg"), "note")
    assert frames.contact_sheet([shot], tmp_path / "sheet.jpg") is None
