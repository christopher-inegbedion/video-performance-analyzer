"""Tests for the parts that don't need TRIBE installed."""

from __future__ import annotations

import numpy as np
import pytest

from vpa.analysis import compare as cmp_mod
from vpa.analysis import features as feat


def make_preds(trace, vertices=32):
    """Synthesise predictions whose magnitude follows a given shape."""
    base = np.ones((len(trace), vertices))
    return base * np.asarray(trace).reshape(-1, 1)


def test_extract_basic_shape():
    preds = make_preds([1, 2, 3, 2, 1])
    f = feat.extract(preds, duration_s=5.0)
    assert f.n_steps == 5
    assert f.duration_s == 5.0
    # Normalised around its own mean. Tolerance accounts for the deliberate
    # 4-decimal rounding in `extract`, which keeps stored traces compact.
    assert abs(np.mean(f.trace) - 1.0) < 1e-3
    assert f.peak_value > 1.0
    assert f.trough_value < 1.0
    assert f.swing > 0


def test_decay_positive_when_front_loaded():
    f = feat.extract(make_preds([5, 5, 3, 1, 1]), duration_s=5.0)
    assert f.decay > 0


def test_decay_negative_when_back_loaded():
    f = feat.extract(make_preds([1, 1, 3, 5, 5]), duration_s=5.0)
    assert f.decay < 0


def test_tail_spike_detected_and_excluded_from_peak():
    # Flat film, then a huge final-step jump — the classic cut-to-black artefact.
    trace = [1.0] * 9 + [8.0]
    f = feat.extract(make_preds(trace), duration_s=10.0, ignore_tail_s=1.5)
    assert f.tail_spike is True
    # The peak must NOT be the artefact at the end.
    assert f.peak_time_s < 9.0


def test_segments_parse_and_verdicts():
    segs = feat.parse_segments("hook:0-2,body:2-8,end:8-10", 10.0)
    assert segs[0] == ("hook", 0.0, 2.0)
    f = feat.extract(
        make_preds([3, 3, 1, 1, 1, 1, 1, 1, 1, 1]),
        duration_s=10.0,
        segments=segs,
    )
    labels = {s.label: s for s in f.segments}
    assert labels["hook"].verdict == "strong"
    assert labels["body"].verdict == "weak"


def test_parse_segments_rejects_garbage():
    with pytest.raises(ValueError):
        feat.parse_segments("hook:notanumber-2", 10.0)


def test_resample_preserves_endpoints():
    out = feat.resample([0.0, 1.0], 5)
    assert len(out) == 5
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_round_trip_features_dict():
    f = feat.extract(make_preds([1, 2, 1]), duration_s=3.0,
                     segments=[("all", 0.0, 3.0)])
    again = feat.Features.from_dict(f.to_dict())
    assert again.n_steps == f.n_steps
    assert again.segments[0].label == "all"


def test_compare_identical_curves_correlate():
    a = feat.extract(make_preds([1, 3, 2, 4, 1]), duration_s=5.0)
    b = feat.extract(make_preds([1, 3, 2, 4, 1]), duration_s=5.0)
    c = cmp_mod.compare(a, b, "ref")
    assert c.correlation > 0.99
    assert c.same_shape is True


def test_compare_inverted_curves_anticorrelate():
    a = feat.extract(make_preds([1, 2, 3, 4, 5]), duration_s=5.0)
    b = feat.extract(make_preds([5, 4, 3, 2, 1]), duration_s=5.0)
    c = cmp_mod.compare(a, b, "ref")
    assert c.correlation < -0.9
    assert c.same_shape is False


def test_compare_handles_different_durations():
    a = feat.extract(make_preds([1, 2, 3]), duration_s=3.0)
    b = feat.extract(make_preds([1, 1, 2, 2, 3, 3, 3, 3]), duration_s=30.0)
    c = cmp_mod.compare(a, b, "long-ref")
    assert len(c.aligned_subject) == len(c.aligned_reference)
    assert any("Durations differ" in n for n in c.notes)


def test_history_position_empty():
    f = feat.extract(make_preds([1, 2, 1]), duration_s=3.0)
    pos = cmp_mod.position_in_history(f, [])
    assert pos.n_previous == 0
    assert pos.notes
