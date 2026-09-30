import json
import subprocess

import pytest

from vpa import fetch


def test_is_url_distinguishes_links_from_paths():
    assert fetch.is_url("https://www.tiktok.com/@x/video/123")
    assert fetch.is_url("http://example.com/v.mp4")
    assert not fetch.is_url("/Users/me/Downloads/clip.mp4")
    assert not fetch.is_url("clip.mp4")
    # A Windows path starts with a letter and a colon, not a scheme.
    assert not fetch.is_url("C:/videos/clip.mp4")


def test_missing_yt_dlp_is_explained_not_raised_raw(monkeypatch):
    """The tool must say what to install, not surface a FileNotFoundError."""
    monkeypatch.setattr(fetch.shutil, "which", lambda _: None)
    with pytest.raises(fetch.FetchError) as exc:
        fetch.probe("https://www.tiktok.com/@x/video/1")
    assert "yt-dlp" in str(exc.value)
    assert "install" in str(exc.value).lower()


def _payload(**over):
    base = {
        "id": "7412345",
        "title": "a clip",
        "uploader": "someone",
        "duration": 12.0,
        "view_count": 759,
        "like_count": 131,
        "comment_count": 4,
        "repost_count": 9,
        "timestamp": 1790769600,   # 2026-09-30 UTC
        "extractor_key": "TikTok",
    }
    base.update(over)
    return base


def _fake_run(payload, returncode=0, stderr=""):
    def run(args, timeout=300):
        return subprocess.CompletedProcess(
            args, returncode, stdout=json.dumps(payload), stderr=stderr
        )
    return run


def test_probe_normalises_platform_metrics(monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    monkeypatch.setattr(fetch, "_run", _fake_run(_payload()))
    info = fetch.probe("https://www.tiktok.com/@x/video/7412345")
    assert info.views == 759
    assert info.likes == 131
    assert info.comments == 4
    # TikTok calls a share a repost; it must land in the shares field.
    assert info.shares == 9
    assert info.platform == "tiktok"
    assert info.posted_at == "2026-09-30"
    assert info.has_metrics()


def test_upload_date_is_used_when_there_is_no_timestamp(monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    monkeypatch.setattr(
        fetch, "_run", _fake_run(_payload(timestamp=None, upload_date="20260930"))
    )
    assert fetch.probe("https://x/1").posted_at == "2026-09-30"


def test_views_without_interactions_does_not_count_as_metrics(monkeypatch):
    """Mirrors the engagement-rate rule: views alone cannot be scored."""
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    monkeypatch.setattr(fetch, "_run", _fake_run(
        _payload(like_count=None, comment_count=None, repost_count=None)))
    info = fetch.probe("https://x/1")
    assert info.views == 759
    assert not info.has_metrics()


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("ERROR: [TikTok] Video is private", "login"),
        ("ERROR: Video unavailable", "unavailable"),
        ("ERROR: Unsupported URL: https://x/1", "not a link"),
    ],
)
def test_failures_are_translated_into_actionable_messages(monkeypatch, stderr, expected):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    monkeypatch.setattr(fetch, "_run", _fake_run({}, returncode=1, stderr=stderr))
    with pytest.raises(fetch.FetchError) as exc:
        fetch.probe("https://x/1")
    assert expected in str(exc.value).lower()


def test_metric_fields_match_the_recorder_signature():
    """Guards the call in the CLI, which splats this straight into metrics.record."""
    import inspect

    from vpa import metrics
    accepted = set(inspect.signature(metrics.record).parameters)
    assert set(fetch.SourceInfo(url="u").metric_fields()) <= accepted


def test_downloaded_path_prefers_the_merged_file(tmp_path):
    """`filename` can name a pre-merge intermediate that no longer exists."""
    merged = tmp_path / "7412345.mp4"
    merged.write_bytes(b"x")
    payload = {
        "id": "7412345",
        "filename": str(tmp_path / "7412345.f303.mp4"),   # gone after merging
        "requested_downloads": [{"filepath": str(merged)}],
    }
    assert fetch._downloaded_path(payload, tmp_path) == merged


def test_downloaded_path_falls_back_to_the_id(tmp_path):
    found = tmp_path / "7412345.mp4"
    found.write_bytes(b"x")
    assert fetch._downloaded_path({"id": "7412345"}, tmp_path) == found


def test_downloaded_path_returns_none_when_nothing_landed(tmp_path):
    assert fetch._downloaded_path({"id": "nope"}, tmp_path) is None
