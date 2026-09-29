"""Storage, metrics ingestion and the learning loop."""

from __future__ import annotations

import csv

import pytest

from vpa import db, learn, metrics


@pytest.fixture()
def store(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "db_path", lambda: path)
    db.init(path)
    return path


def _video(store, label, **kw):
    return db.add_video({"path": f"/tmp/{label}.mp4", "label": label,
                         "duration_s": 24.0, "sha1": label, **kw}, store)


def test_video_roundtrip(store):
    vid = _video(store, "cut-a")
    row = db.get_video(vid, store)
    assert row["label"] == "cut-a"
    assert db.find_video_by_sha1("cut-a", store)["id"] == vid


def test_evaluation_lifecycle(store):
    vid = _video(store, "cut-a")
    eid = db.create_evaluation(vid, None, "facebook/tribev2-mini", store)
    assert db.get_evaluation(eid, store)["status"] == "queued"
    db.update_evaluation(eid, path=store, status="done", progress=1.0,
                         features={"opening_2s": 1.1})
    row = db.get_evaluation(eid, store)
    assert row["status"] == "done"
    assert '"opening_2s"' in row["features"]


def test_resolve_by_prefix(store):
    vid = _video(store, "cut-a")
    eid = db.create_evaluation(vid, None, "ckpt", store)
    assert db.resolve_evaluation(eid[:8], store)["id"] == eid
    assert db.resolve_evaluation("nope_", store) is None


def test_recommendation_generations_increment(store):
    vid = _video(store, "cut-a")
    eid = db.create_evaluation(vid, None, "ckpt", store)
    db.add_recommendation(eid, "first", "m", "initial", path=store)
    db.add_recommendation(eid, "second", "m", "retroactive", path=store)
    recs = db.recommendations_for(eid, store)
    assert [r["generation"] for r in recs] == [1, 2]
    assert recs[1]["kind"] == "retroactive"


def test_metrics_record_and_resolve_by_label(store, monkeypatch):
    monkeypatch.setattr(db, "db_path", lambda: store)
    _video(store, "cut-a")
    metrics.record("cut-a", views=1000, likes=50, path=store)
    vid = metrics.resolve_video("cut-a", store)
    assert len(db.performance_for(vid, store)) == 1


def test_metrics_unknown_video_raises(store):
    with pytest.raises(ValueError, match="No registered video"):
        metrics.record("ghost", views=1, path=store)


def test_engagement_rate_prefers_ratio():
    assert learn.engagement_rate(
        {"views": 1000, "likes": 50, "comments": 0, "shares": 0, "saves": 0}
    ) == pytest.approx(0.05)
    # No views recorded: fall back to raw interactions rather than inventing a rate
    assert learn.engagement_rate(
        {"views": None, "likes": 7, "comments": 0, "shares": 0, "saves": 0}
    ) == 7.0
    assert learn.engagement_rate(
        {"views": None, "likes": None, "comments": None, "shares": None, "saves": None}
    ) is None


def test_learning_is_honest_about_small_samples(store, monkeypatch):
    monkeypatch.setattr(db, "db_path", lambda: store)
    for i in range(3):
        vid = _video(store, f"v{i}")
        eid = db.create_evaluation(vid, None, "ckpt", store)
        db.update_evaluation(eid, path=store, status="done",
                             features={"opening_2s": 1.0 + i * 0.1, "closing_2s": 0.9,
                                       "peak_value": 1.2, "trough_value": 0.8,
                                       "swing": 0.4, "variability": 0.1,
                                       "decay": 0.1, "sustained_above": 0.5})
        db.add_performance({"video_id": vid, "views": 1000, "likes": 10 * (i + 1)}, store)
    model = learn.build(store)
    assert model.n_labelled == 3
    assert model.usable is False          # 3 < MIN_FOR_SIGNAL
    assert "anecdote" in model.caveat


def test_csv_import(store, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "db_path", lambda: store)
    _video(store, "cut-a")
    csv_path = tmp_path / "m.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "views", "likes", "watch_through"])
        w.writerow(["cut-a", "12.4k", "380", "38%"])
        w.writerow(["missing-video", "1", "1", ""])
    n, problems = metrics.import_csv(csv_path, store)
    assert n == 1
    assert len(problems) == 1
    perf = db.performance_for(metrics.resolve_video("cut-a", store), store)
    assert perf[0]["views"] == 12400
    assert perf[0]["watch_through"] == pytest.approx(0.38)


def test_csv_requires_video_column(store, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("views,likes\n1,2\n")
    with pytest.raises(ValueError, match="needs a 'video' column"):
        metrics.import_csv(bad, store)
