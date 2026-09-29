"""Orchestration: register a video, score it, analyse it, store everything.

Kept separate from the CLI so the same flow can be driven from a script, a test,
or a future web UI without touching argument parsing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db, learn, tribe
from . import frames as frames_mod
from .analysis import compare as cmp_mod
from .analysis import features as feat_mod
from .config import Config, data_dir

ProgressFn = Callable[[str, float, str], None]


def sha1_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hash the first and last megabyte plus size — fast, and good enough to spot
    the same file re-registered under a different name."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk * 2:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()


def register_video(
    path: Path, label: str | None = None, kind: str = "subject", reuse: bool = True
) -> str:
    """Add a video to the store, reusing the row if we've seen this file before."""
    path = path.expanduser().resolve()
    info = tribe.probe(path)
    digest = sha1_of(path)
    if reuse and (existing := db.find_video_by_sha1(digest)):
        return existing["id"]
    return db.add_video(
        {
            "path": path,
            "label": label or path.stem,
            "kind": kind,
            "duration_s": info.duration_s,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "has_audio": info.has_audio,
            "sha1": digest,
        }
    )


def preds_dir() -> Path:
    d = data_dir() / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir(eval_id: str) -> Path:
    d = data_dir() / "work" / eval_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_features_for_video(video_id: str) -> feat_mod.Features | None:
    """Reuse a previous successful scoring of the same file rather than re-running
    a job that can take many minutes."""
    for row in db.list_evaluations(limit=200, status="done"):
        if row["video_id"] == video_id and row["features"]:
            try:
                return feat_mod.Features.from_dict(json.loads(row["features"]))
            except (json.JSONDecodeError, TypeError):
                continue
    return None


@dataclass
class EvaluationResult:
    evaluation_id: str
    features: feat_mod.Features
    comparison: cmp_mod.Comparison | None
    history: cmp_mod.HistoryPosition | None
    learned: learn.LearnedModel
    notes: list[str]
    video_label: str
    key_frames: list[Any] = field(default_factory=list)
    contact_sheet: str | None = None


def evaluate(
    cfg: Config,
    video_path: Path,
    reference_path: Path | None = None,
    label: str | None = None,
    reference_label: str | None = None,
    segments_spec: str | None = None,
    n_auto_segments: int = 4,
    extract_frames: bool = True,
    progress: ProgressFn | None = None,
    reuse_reference: bool = True,
) -> EvaluationResult:
    """The full flow: score, analyse, compare, persist."""
    notes: list[str] = []

    video_id = register_video(video_path, label=label, kind="subject")
    ref_id = None
    if reference_path:
        ref_id = register_video(reference_path, label=reference_label, kind="reference")

    eval_id = db.create_evaluation(video_id, ref_id, cfg.tribe.checkpoint)
    vrow = db.get_video(video_id)
    video_label = vrow["label"] or Path(vrow["path"]).stem

    def tick(stage: str, frac: float, detail: str = "") -> None:
        db.update_evaluation(eval_id, status=stage, progress=frac, stage_detail=detail)
        if progress:
            progress(stage, frac, detail)

    try:
        import numpy as np

        db.update_evaluation(eval_id, started_at=__import__("time").time())

        preds, modalities, run_notes = tribe.run(
            Path(vrow["path"]), work_dir(eval_id), cfg, progress=tick
        )
        notes.extend(run_notes)

        preds_path = preds_dir() / f"{eval_id}.npy"
        np.save(preds_path, preds)

        duration = float(vrow["duration_s"] or 0.0)
        if segments_spec:
            segs = feat_mod.parse_segments(segments_spec, duration)
        else:
            segs = feat_mod.auto_segments(duration, n_auto_segments)
            notes.append(
                "Sections were split evenly because none were given. Use "
                "--segments \"hook:0-3,body:3-16,end:16-24\" to match your real structure."
            )

        features = feat_mod.extract(
            preds,
            duration_s=duration,
            modalities=modalities,
            strong=cfg.analysis.strong_threshold,
            weak=cfg.analysis.weak_threshold,
            ignore_tail_s=cfg.analysis.ignore_tail_s,
            segments=segs,
        )

        # --- 1:1 comparison against a reference, if one was given
        comparison = None
        if ref_id:
            rrow = db.get_video(ref_id)
            ref_feats = latest_features_for_video(ref_id) if reuse_reference else None
            if ref_feats is None:
                tick("encoding-reference", 0.6, "scoring the reference video")
                rpreds, rmods, rnotes = tribe.run(
                    Path(rrow["path"]), work_dir(eval_id) / "ref", cfg, progress=None
                )
                notes.extend(f"reference: {n}" for n in rnotes)
                rdur = float(rrow["duration_s"] or 0.0)
                ref_feats = feat_mod.extract(
                    rpreds,
                    duration_s=rdur,
                    modalities=rmods,
                    strong=cfg.analysis.strong_threshold,
                    weak=cfg.analysis.weak_threshold,
                    ignore_tail_s=cfg.analysis.ignore_tail_s,
                    segments=feat_mod.auto_segments(rdur, n_auto_segments),
                )
                # Persist the reference scoring as its own evaluation so it is
                # never re-computed.
                ref_eval = db.create_evaluation(ref_id, None, cfg.tribe.checkpoint)
                db.update_evaluation(
                    ref_eval, status="done", progress=1.0,
                    features=ref_feats.to_dict(), modalities=rmods,
                    finished_at=__import__("time").time(),
                )
            else:
                notes.append("Reused an earlier scoring of the reference video.")
            comparison = cmp_mod.compare(
                features,
                ref_feats,
                rrow["label"] or Path(rrow["path"]).stem,
                same_shape_r=cfg.analysis.same_shape_r,
            )

        # --- the actual frames at the moments the curve flags
        key_frames: list[Any] = []
        sheet: str | None = None
        if extract_frames:
            frame_dir = work_dir(eval_id) / "frames"
            key_frames = frames_mod.extract(Path(vrow["path"]), features, frame_dir)
            if key_frames:
                made = frames_mod.contact_sheet(key_frames, frame_dir / "contact-sheet.jpg")
                sheet = str(made) if made else None

        # --- position within the user's own history
        hist_entries = [
            h for h in learn.history_for_comparison() if h["evaluation_id"] != eval_id
        ]
        history = cmp_mod.position_in_history(features, hist_entries)
        learned = learn.build()

        db.update_evaluation(
            eval_id,
            status="done",
            progress=1.0,
            stage_detail="",
            modalities=modalities,
            preds_path=str(preds_path),
            features=features.to_dict(),
            comparison=comparison.to_dict() if comparison else None,
            frames=[k.to_dict() for k in key_frames],
            finished_at=__import__("time").time(),
        )
        return EvaluationResult(
            eval_id, features, comparison, history, learned, notes, video_label,
            key_frames=key_frames, contact_sheet=sheet,
        )

    except Exception as exc:
        db.update_evaluation(
            eval_id, status="failed", error=str(exc),
            finished_at=__import__("time").time()
        )
        raise


def load_frames(eval_id: str) -> list[dict[str, Any]]:
    """Key frames stored with an evaluation, if it was scored with them."""
    row = db.get_evaluation(eval_id)
    if not row:
        return []
    with contextlib.suppress(json.JSONDecodeError, TypeError, IndexError, KeyError):
        if row["frames"]:
            return json.loads(row["frames"])
    return []


def load_result(eval_id: str) -> tuple[feat_mod.Features | None, cmp_mod.Comparison | None, Any]:
    """Rehydrate a stored evaluation."""
    row = db.get_evaluation(eval_id)
    if not row:
        return None, None, None
    features = None
    comparison = None
    if row["features"]:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            features = feat_mod.Features.from_dict(json.loads(row["features"]))
    if row["comparison"]:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d = json.loads(row["comparison"])
            d["sections"] = [cmp_mod.SectionDelta(**s) for s in d.get("sections", [])]
            comparison = cmp_mod.Comparison(**d)
    return features, comparison, row
