"""SQLite storage for evaluations, features, metrics, recommendations and chat.

One file, no server, easy to inspect with any sqlite3 client. Schema is
versioned so upgrades don't lose history.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import data_dir

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- A video the user has registered. Either their own cut or a reference.
CREATE TABLE IF NOT EXISTS videos (
    id           TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    label        TEXT,
    kind         TEXT NOT NULL DEFAULT 'subject',   -- subject | reference
    duration_s   REAL,
    width        INTEGER,
    height       INTEGER,
    fps          REAL,
    has_audio    INTEGER NOT NULL DEFAULT 0,
    sha1         TEXT,
    source_url   TEXT,                                -- where it was fetched from
    created_at   REAL NOT NULL
);

-- One TRIBE run over one video.
CREATE TABLE IF NOT EXISTS evaluations (
    id            TEXT PRIMARY KEY,
    video_id      TEXT NOT NULL REFERENCES videos(id),
    reference_id  TEXT REFERENCES videos(id),
    status        TEXT NOT NULL,        -- queued|extracting|encoding|predicting|done|failed
    stage_detail  TEXT,
    progress      REAL NOT NULL DEFAULT 0.0,
    modalities    TEXT,                 -- json list, e.g. ["Video","Audio"]
    checkpoint    TEXT,
    preds_path    TEXT,
    features      TEXT,                 -- json blob from analysis.features
    comparison    TEXT,                 -- json blob from analysis.compare
    frames        TEXT,                 -- json list of extracted key frames
    error         TEXT,
    started_at    REAL,
    finished_at   REAL,
    created_at    REAL NOT NULL
);

-- Real-world outcome for a published video. This is what makes the tool learn.
CREATE TABLE IF NOT EXISTS performance (
    id            TEXT PRIMARY KEY,
    video_id      TEXT NOT NULL REFERENCES videos(id),
    platform      TEXT,
    posted_at     REAL,
    views         INTEGER,
    likes         INTEGER,
    comments      INTEGER,
    shares        INTEGER,
    saves         INTEGER,
    watch_through REAL,                 -- 0..1 if known
    followers_at_post INTEGER,
    notes         TEXT,
    recorded_at   REAL NOT NULL
);

-- LLM output. Multiple generations per evaluation are kept, so retroactive
-- passes append rather than overwrite — you can see how advice changed.
CREATE TABLE IF NOT EXISTS recommendations (
    id             TEXT PRIMARY KEY,
    evaluation_id  TEXT NOT NULL REFERENCES evaluations(id),
    generation     INTEGER NOT NULL DEFAULT 1,
    kind           TEXT NOT NULL DEFAULT 'initial',  -- initial | retroactive
    model          TEXT,
    body           TEXT NOT NULL,       -- markdown
    evidence       TEXT,                -- json: what data informed this
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chat (
    id             TEXT PRIMARY KEY,
    evaluation_id  TEXT REFERENCES evaluations(id),
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    created_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_video ON evaluations(video_id);
CREATE INDEX IF NOT EXISTS idx_perf_video ON performance(video_id);
CREATE INDEX IF NOT EXISTS idx_rec_eval   ON recommendations(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_chat_eval  ON chat(evaluation_id);
"""


def db_path() -> Path:
    return data_dir() / "vpa.db"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations only. A user's evaluation history is not disposable."""
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(evaluations)")}
    if "frames" not in columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN frames TEXT")
    vcols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    if "source_url" not in vcols:
        conn.execute("ALTER TABLE videos ADD COLUMN source_url TEXT")


def init(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )


# ---------------------------------------------------------------- videos


def add_video(row: dict[str, Any], path: Path | None = None) -> str:
    vid = row.get("id") or new_id("vid")
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO videos(id, path, label, kind, duration_s, width, height,
                                  fps, has_audio, sha1, source_url, created_at)
               VALUES(:id,:path,:label,:kind,:duration_s,:width,:height,:fps,
                      :has_audio,:sha1,:source_url,:created_at)""",
            {
                "id": vid,
                "path": str(row["path"]),
                "label": row.get("label"),
                "kind": row.get("kind", "subject"),
                "duration_s": row.get("duration_s"),
                "width": row.get("width"),
                "height": row.get("height"),
                "fps": row.get("fps"),
                "has_audio": int(bool(row.get("has_audio"))),
                "sha1": row.get("sha1"),
                "source_url": row.get("source_url"),
                "created_at": time.time(),
            },
        )
    return vid


def find_video_by_sha1(sha1: str, path: Path | None = None) -> sqlite3.Row | None:
    with connect(path) as conn:
        return conn.execute("SELECT * FROM videos WHERE sha1=?", (sha1,)).fetchone()


def set_source_url(vid: str, url: str, path: Path | None = None) -> None:
    """Attach a source link to a video we already had on disk.

    A file can be registered first and recognised as a published post later;
    recording the link then is what lets `metrics sync` refresh its numbers.
    """
    with connect(path) as conn:
        conn.execute("UPDATE videos SET source_url=? WHERE id=?", (url, vid))


def get_video(vid: str, path: Path | None = None) -> sqlite3.Row | None:
    with connect(path) as conn:
        return conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()


def list_videos(kind: str | None = None, path: Path | None = None) -> list[sqlite3.Row]:
    q = "SELECT * FROM videos"
    args: tuple = ()
    if kind:
        q += " WHERE kind=?"
        args = (kind,)
    q += " ORDER BY created_at DESC"
    with connect(path) as conn:
        return conn.execute(q, args).fetchall()


# ------------------------------------------------------------ evaluations


def create_evaluation(
    video_id: str, reference_id: str | None, checkpoint: str, path: Path | None = None
) -> str:
    eid = new_id("eval")
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO evaluations(id, video_id, reference_id, status, progress,
                                       checkpoint, created_at)
               VALUES(?,?,?,'queued',0.0,?,?)""",
            (eid, video_id, reference_id, checkpoint, time.time()),
        )
    return eid


def update_evaluation(eid: str, path: Path | None = None, **fields: Any) -> None:
    if not fields:
        return
    for key in ("features", "comparison", "modalities", "frames"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key])
    sets = ", ".join(f"{k}=:{k}" for k in fields)
    fields["id"] = eid
    with connect(path) as conn:
        conn.execute(f"UPDATE evaluations SET {sets} WHERE id=:id", fields)


def get_evaluation(eid: str, path: Path | None = None) -> sqlite3.Row | None:
    with connect(path) as conn:
        return conn.execute("SELECT * FROM evaluations WHERE id=?", (eid,)).fetchone()


def list_evaluations(
    limit: int = 50, status: str | None = None, path: Path | None = None
) -> list[sqlite3.Row]:
    q = """SELECT e.*, v.label AS video_label, v.path AS video_path,
                  r.label AS reference_label
           FROM evaluations e
           JOIN videos v ON v.id = e.video_id
           LEFT JOIN videos r ON r.id = e.reference_id"""
    args: list = []
    if status:
        q += " WHERE e.status=?"
        args.append(status)
    q += " ORDER BY e.created_at DESC LIMIT ?"
    args.append(limit)
    with connect(path) as conn:
        return conn.execute(q, args).fetchall()


def resolve_evaluation(prefix: str, path: Path | None = None) -> sqlite3.Row | None:
    """Accept a short id prefix so users don't type the whole thing."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluations WHERE id LIKE ? ORDER BY created_at DESC",
            (f"{prefix}%",),
        ).fetchall()
    if len(rows) == 1:
        return rows[0]
    if not rows:
        return None
    raise ValueError(f"'{prefix}' matches {len(rows)} evaluations; be more specific")


# ----------------------------------------------------------- performance


def add_performance(row: dict[str, Any], path: Path | None = None) -> str:
    pid = new_id("perf")
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO performance(id, video_id, platform, posted_at, views, likes,
                                       comments, shares, saves, watch_through,
                                       followers_at_post, notes, recorded_at)
               VALUES(:id,:video_id,:platform,:posted_at,:views,:likes,:comments,
                      :shares,:saves,:watch_through,:followers_at_post,:notes,:recorded_at)""",
            {
                "id": pid,
                "video_id": row["video_id"],
                "platform": row.get("platform"),
                "posted_at": row.get("posted_at"),
                "views": row.get("views"),
                "likes": row.get("likes"),
                "comments": row.get("comments"),
                "shares": row.get("shares"),
                "saves": row.get("saves"),
                "watch_through": row.get("watch_through"),
                "followers_at_post": row.get("followers_at_post"),
                "notes": row.get("notes"),
                "recorded_at": time.time(),
            },
        )
    return pid


def performance_for(video_id: str, path: Path | None = None) -> list[sqlite3.Row]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT * FROM performance WHERE video_id=? ORDER BY recorded_at DESC",
            (video_id,),
        ).fetchall()


def labelled_history(path: Path | None = None) -> list[sqlite3.Row]:
    """Every completed evaluation that has a real-world outcome attached.

    This join is the whole self-improvement mechanism: predicted features on one
    side, what actually happened on the other.
    """
    with connect(path) as conn:
        return conn.execute(
            """SELECT e.id AS evaluation_id, e.features, e.created_at,
                      v.id AS video_id, v.label, v.duration_s,
                      p.views, p.likes, p.comments, p.shares, p.saves,
                      p.watch_through, p.platform, p.followers_at_post
               FROM evaluations e
               JOIN videos v ON v.id = e.video_id
               JOIN performance p ON p.video_id = v.id
               WHERE e.status='done' AND e.features IS NOT NULL
               ORDER BY e.created_at"""
        ).fetchall()


# -------------------------------------------------------- recommendations


def add_recommendation(
    evaluation_id: str,
    body: str,
    model: str,
    kind: str = "initial",
    evidence: dict | None = None,
    path: Path | None = None,
) -> str:
    rid = new_id("rec")
    with connect(path) as conn:
        gen = conn.execute(
            "SELECT COALESCE(MAX(generation),0)+1 FROM recommendations WHERE evaluation_id=?",
            (evaluation_id,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO recommendations(id, evaluation_id, generation, kind, model,
                                           body, evidence, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                rid,
                evaluation_id,
                gen,
                kind,
                model,
                body,
                json.dumps(evidence or {}),
                time.time(),
            ),
        )
    return rid


def recommendations_for(evaluation_id: str, path: Path | None = None) -> list[sqlite3.Row]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT * FROM recommendations WHERE evaluation_id=? ORDER BY generation",
            (evaluation_id,),
        ).fetchall()


# ------------------------------------------------------------------ chat


def add_chat(evaluation_id: str | None, role: str, content: str, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO chat(id, evaluation_id, role, content, created_at) VALUES(?,?,?,?,?)",
            (new_id("msg"), evaluation_id, role, content, time.time()),
        )


def chat_history(
    evaluation_id: str | None, limit: int = 40, path: Path | None = None
) -> list[sqlite3.Row]:
    with connect(path) as conn:
        if evaluation_id:
            rows = conn.execute(
                """SELECT * FROM chat WHERE evaluation_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (evaluation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat WHERE evaluation_id IS NULL ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return list(reversed(rows))
