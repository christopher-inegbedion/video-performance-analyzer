"""Pull a video, and what it did, straight from the platform that hosts it.

The point of this module is not convenience downloading. A published video
already carries the outcome data this tool wants to learn from — views, likes,
comments, shares — and typing those in by hand means they get entered once, at
whatever moment the user happened to look, and never corrected. Reading them
from the source makes the number reproducible and re-pullable, which is what
turns a pile of one-off entries into a series you can actually fit against.

yt-dlp is an optional dependency. Everything here degrades to a clear message
rather than a traceback when it is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Platforms whose stats we know how to read. Anything else still downloads;
# we just don't claim to understand its numbers.
_KNOWN = {
    "tiktok": "tiktok",
    "youtube": "youtube",
    "instagram": "instagram",
}


class FetchError(RuntimeError):
    """Raised when a URL cannot be turned into a local video."""


@dataclass
class SourceInfo:
    """What the platform says about a video, normalised across sites."""

    url: str
    platform: str | None = None
    title: str | None = None
    uploader: str | None = None
    duration_s: float | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    posted_at: str | None = None       # YYYY-MM-DD

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metric_fields(self) -> dict[str, Any]:
        """Just the performance numbers, for handing to the metrics store."""
        return {
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "platform": self.platform,
            "posted_at": self.posted_at,
        }

    def has_metrics(self) -> bool:
        """True when the platform gave us at least one interaction count.

        Views alone are not enough: an entry with no interactions cannot be
        scored, so claiming we captured performance would be misleading.
        """
        return any(v is not None for v in (self.likes, self.comments, self.shares))


def is_url(value: str) -> bool:
    """Cheap check so the CLI can accept a path or a link in the same argument."""
    return value.startswith(("http://", "https://"))


def available() -> bool:
    return shutil.which("yt-dlp") is not None


def _require() -> None:
    if not available():
        raise FetchError(
            "yt-dlp is not installed, so links cannot be fetched.\n"
            "  install it with:  pip install 'video-performance-analyzer[fetch]'\n"
            "  or:               brew install yt-dlp\n"
            "You can still analyse a downloaded file by passing its path."
        )


def _platform_of(extractor: str | None, url: str) -> str | None:
    hay = f"{extractor or ''} {url}".lower()
    for needle, name in _KNOWN.items():
        if needle in hay:
            return name
    return None


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:                      # yt-dlp vanished mid-run
        raise FetchError("yt-dlp is not installed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"yt-dlp timed out after {timeout}s on {args[-1]}") from exc


def _parse(payload: dict[str, Any], url: str) -> SourceInfo:
    posted = None
    if ts := payload.get("timestamp"):
        try:
            posted = datetime.fromtimestamp(int(ts), tz=UTC).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            posted = None
    # yt-dlp's fallback format is YYYYMMDD.
    d = str(payload.get("upload_date") or "")
    if not posted and len(d) == 8 and d.isdigit():
        posted = f"{d[:4]}-{d[4:6]}-{d[6:]}"

    return SourceInfo(
        url=url,
        platform=_platform_of(payload.get("extractor_key") or payload.get("extractor"), url),
        title=payload.get("title") or payload.get("description"),
        uploader=payload.get("uploader") or payload.get("creator") or payload.get("channel"),
        duration_s=payload.get("duration"),
        views=payload.get("view_count"),
        likes=payload.get("like_count"),
        comments=payload.get("comment_count"),
        # TikTok calls a share a repost; other sites use repost_count the same way.
        shares=payload.get("repost_count"),
        posted_at=posted,
    )


def probe(url: str, timeout: int = 120) -> SourceInfo:
    """Read a video's metadata without downloading it.

    Used by `metrics sync`, where the file is already on disk and only the
    numbers need refreshing.
    """
    _require()
    proc = _run(["yt-dlp", "--dump-single-json", "--no-warnings",
                 "--no-playlist", url], timeout=timeout)
    if proc.returncode != 0:
        raise FetchError(_explain(proc.stderr, url))
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FetchError(f"yt-dlp returned something that was not JSON for {url}") from exc
    return _parse(payload, url)


def fetch(url: str, dest_dir: Path, timeout: int = 600) -> tuple[Path, SourceInfo]:
    """Download a video and return its local path alongside its stats."""
    _require()
    dest_dir.mkdir(parents=True, exist_ok=True)
    template = str(dest_dir / "%(id)s.%(ext)s")

    proc = _run([
        "yt-dlp",
        "--no-warnings", "--no-playlist",
        # Merge to mp4 so the rest of the pipeline sees the container it expects.
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "--print-json", "--no-simulate",
        "-o", template,
        url,
    ], timeout=timeout)
    if proc.returncode != 0:
        raise FetchError(_explain(proc.stderr, url))

    # --print-json emits one object per downloaded item; we disabled playlists
    # so there is exactly one, but take the last line defensively.
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise FetchError(f"yt-dlp downloaded nothing for {url}")
    payload = json.loads(lines[-1])
    info = _parse(payload, url)

    path = _downloaded_path(payload, dest_dir)
    if path is None:
        raise FetchError(f"yt-dlp reported success but no file appeared for {url}")
    return path, info


def _downloaded_path(payload: dict[str, Any], dest_dir: Path) -> Path | None:
    """Find the file yt-dlp actually wrote.

    `filename` is the pre-merge name and can name an intermediate that no
    longer exists once streams are combined, so prefer what the downloader
    reports last and fall back to matching on the video id.
    """
    for candidate in (
        (payload.get("requested_downloads") or [{}])[-1].get("filepath"),
        payload.get("filepath"),
        payload.get("_filename"),
        payload.get("filename"),
    ):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    if vid := payload.get("id"):
        matches = sorted(dest_dir.glob(f"{vid}.*"))
        if matches:
            return matches[0]
    return None


def _explain(stderr: str, url: str) -> str:
    """Turn yt-dlp's noise into something a user can act on."""
    low = (stderr or "").lower()
    if "private" in low or "login" in low or "sign in" in low:
        return f"{url} needs a login — yt-dlp cannot reach it anonymously."
    if "unavailable" in low or "not exist" in low or "404" in low:
        return f"{url} is unavailable or has been removed."
    if "unsupported url" in low:
        return f"{url} is not a link yt-dlp knows how to read."
    if "rate" in low and "limit" in low:
        return f"the platform is rate-limiting this download ({url}). Try again shortly."
    tail = (stderr or "").strip().splitlines()
    return f"could not fetch {url}" + (f": {tail[-1]}" if tail else "")
