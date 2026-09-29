"""Ingest real-world performance: manual entry and CSV import.

Deliberately platform-agnostic. No OAuth, no developer app, no API review — it
works for every platform including ones with no public API, and it works for a
video you posted to three places at once.

CSV columns (header required, order irrelevant, unknown columns ignored):
    video      label or id of an already-registered video  [required]
    platform   instagram | tiktok | youtube | ...
    posted_at  ISO date or datetime
    views, likes, comments, shares, saves    integers
    watch_through    0..1 or a percentage like 42%
    followers        follower count at time of posting
    notes            free text
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from . import db

REQUIRED = "video"
_INT_FIELDS = ("views", "likes", "comments", "shares", "saves", "followers")


def _parse_int(v: str | None) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().replace(",", "").replace(" ", "")
    mult = 1.0
    if s and s[-1].lower() in {"k", "m"}:
        mult = 1_000 if s[-1].lower() == "k" else 1_000_000
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return None


def _parse_rate(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        f = float(s)
        return f / 100.0 if f > 1.0 else f
    except ValueError:
        return None


def _parse_when(v: str | None) -> float | None:
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def resolve_video(token: str, path: Path | None = None) -> str | None:
    """Accept a video id, an id prefix, or an exact label."""
    with db.connect(path) as conn:
        row = conn.execute("SELECT id FROM videos WHERE id=?", (token,)).fetchone()
        if row:
            return row["id"]
        rows = conn.execute("SELECT id FROM videos WHERE label=?", (token,)).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        if len(rows) > 1:
            raise ValueError(f"'{token}' matches {len(rows)} videos by label; use the id")
        rows = conn.execute("SELECT id FROM videos WHERE id LIKE ?", (f"{token}%",)).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        if len(rows) > 1:
            raise ValueError(f"'{token}' matches {len(rows)} video ids; be more specific")
    return None


def record(
    video: str,
    platform: str | None = None,
    views: int | None = None,
    likes: int | None = None,
    comments: int | None = None,
    shares: int | None = None,
    saves: int | None = None,
    watch_through: float | None = None,
    followers: int | None = None,
    posted_at: str | None = None,
    notes: str | None = None,
    path: Path | None = None,
) -> str:
    vid = resolve_video(video, path)
    if not vid:
        raise ValueError(
            f"No registered video matches '{video}'. Run `vpa list videos` to see them, "
            "or analyse the video first."
        )
    return db.add_performance(
        {
            "video_id": vid,
            "platform": platform,
            "posted_at": _parse_when(posted_at),
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "saves": saves,
            "watch_through": watch_through,
            "followers_at_post": followers,
            "notes": notes,
        },
        path,
    )


def import_csv(csv_path: Path, path: Path | None = None) -> tuple[int, list[str]]:
    """Returns (rows_imported, problems)."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    imported = 0
    problems: list[str] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or REQUIRED not in [f.lower() for f in reader.fieldnames]:
            raise ValueError(
                f"CSV needs a '{REQUIRED}' column. Found: {', '.join(reader.fieldnames or [])}"
            )
        for i, raw in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): v for k, v in raw.items()}
            token = (row.get("video") or "").strip()
            if not token:
                problems.append(f"line {i}: empty 'video' column, skipped")
                continue
            try:
                record(
                    video=token,
                    platform=row.get("platform"),
                    views=_parse_int(row.get("views")),
                    likes=_parse_int(row.get("likes")),
                    comments=_parse_int(row.get("comments")),
                    shares=_parse_int(row.get("shares")),
                    saves=_parse_int(row.get("saves")),
                    watch_through=_parse_rate(row.get("watch_through")),
                    followers=_parse_int(row.get("followers")),
                    posted_at=row.get("posted_at"),
                    notes=row.get("notes"),
                    path=path,
                )
                imported += 1
            except ValueError as exc:
                problems.append(f"line {i}: {exc}")
    return imported, problems


def write_template(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "platform", "posted_at", "views", "likes",
                    "comments", "shares", "saves", "watch_through", "followers", "notes"])
        w.writerow(["my-cut-v3", "instagram", "2026-09-20", "12400", "380",
                    "24", "11", "63", "38%", "5200", "posted 7pm"])
    return target
