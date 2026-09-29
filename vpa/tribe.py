"""Run Meta's TRIBE v2 brain-encoding model over a video.

TRIBE ships configured for Meta's GPU cluster and does not work out of the box on
a laptop. Everything in this module exists because of a specific failure:

  1. `device: cuda` is baked into the published checkpoint config. On a machine
     without CUDA this raises "Torch not compiled with CUDA enabled".
  2. `num_workers: 20` is also baked in. On macOS the DataLoader workers die
     silently and the process sits at ~3% CPU looking alive forever.
  3. `compute_type` is hardcoded to float16 in TRIBE's whisperx call. CPUs cannot
     do float16, so speech extraction always fails.
  4. Even with that fixed, `uvx whisperx` resolves a torchaudio that is too new
     for it ("module 'torchaudio' has no attribute 'list_audio_backends'").
     We sidestep it entirely: TRIBE reads a cached word-timing .tsv next to the
     extracted wav if one exists, so we generate that ourselves with
     faster-whisper and TRIBE never invokes whisperx at all.
  5. The language pathway loads meta-llama/Llama-3.2-1B, a GATED repo. Without a
     HuggingFace token it 401s. An empty transcript keeps vision + audition
     working with no gate, which is the default here.
"""

from __future__ import annotations

import contextlib
import csv
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config

ProgressFn = Callable[[str, float, str], None]  # stage, 0..1, detail


class TribeError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


# ------------------------------------------------------------------ ffprobe


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise TribeError(
                f"{tool} not found on PATH. Install it first — on macOS: brew install ffmpeg"
            )


def probe(path: Path) -> VideoInfo:
    """Read basic stream facts. Cheap, and needed before anything else."""
    require_ffmpeg()
    if not path.exists():
        raise TribeError(f"No such video: {path}")
    res = _run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    if res.returncode != 0:
        raise TribeError(f"ffprobe failed on {path.name}:\n{res.stderr.strip()}")
    meta = json.loads(res.stdout)
    streams = meta.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise TribeError(f"{path.name} has no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return VideoInfo(
        path=path,
        duration_s=float(meta.get("format", {}).get("duration", 0.0)),
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps=fps,
        has_audio=has_audio,
    )


# ------------------------------------------------------------ preparation


def prepare(src: Path, workdir: Path, cfg: Config, keep_audio: bool = True) -> Path:
    """Downscale and normalise frame rate before scoring.

    Two reasons this matters enormously for runtime:
      * The video encoder normalises to 256px regardless, so a 4K source is
        wasted work.
      * The encoder consumes fixed-length frame clips, so a 60fps source costs
        roughly 2.5x a 24fps one for identical footage. Normalising frame rate is
        also what makes two evaluations comparable to each other.
    """
    require_ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "prepared.mp4"
    vf = f"scale={cfg.tribe.scale_width}:{cfg.tribe.scale_height}"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(src),
        "-vf", vf, "-r", str(cfg.tribe.target_fps),
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
    ]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if keep_audio else ["-an"]
    cmd += [str(out), "-y"]
    res = _run(cmd)
    if res.returncode != 0 or not out.exists():
        raise TribeError(f"ffmpeg could not prepare {src.name}:\n{res.stderr.strip()}")
    return out


def extract_wav(video: Path, dest: Path) -> Path | None:
    """16k mono wav — the format TRIBE's own audio extractor produces."""
    res = _run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ar", "16000",
         "-ac", "1", str(dest), "-y"]
    )
    return dest if res.returncode == 0 and dest.exists() else None


def write_word_timings(wav: Path, model_name: str = "base.en", enable: bool = True) -> Path:
    """Write the .tsv TRIBE looks for, so it never calls whisperx.

    TRIBE caches transcripts as `<wav stem>.tsv` beside the wav and reads them if
    present. We write that file ourselves.

    With `enable=False` we write a header-only file: valid, but producing zero
    Word events, which keeps the gated Llama language encoder out of the graph.
    """
    tsv = wav.with_suffix(".tsv")
    fields = ["text", "start", "duration", "sequence_id", "sentence"]
    rows: list[dict] = []
    if enable:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise TribeError(
                "faster-whisper is required for the language pathway. "
                "Install it with: pip install 'video-performance-analyzer[tribe]'"
            ) from exc
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(wav), beam_size=1, word_timestamps=True, vad_filter=False
        )
        for i, seg in enumerate(segments):
            sentence = seg.text.replace('"', "").strip()
            for w in seg.words or []:
                if w.start is None:
                    continue
                rows.append(
                    {
                        "text": w.word.replace('"', "").strip(),
                        "start": round(w.start, 3),
                        "duration": round(w.end - w.start, 3),
                        "sequence_id": i,
                        "sentence": sentence,
                    }
                )
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return tsv


# ------------------------------------------------------- checkpoint repair


def patch_checkpoint_config(checkpoint: str, cache_dir: Path | None, device: str) -> list[str]:
    """Rewrite cluster-specific settings in the downloaded checkpoint config.

    Returns a list of human-readable changes so the tool can tell the user what
    it had to do rather than silently mutating a cached file.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TribeError(
            "huggingface_hub missing. Install with: pip install "
            "'video-performance-analyzer[tribe]'"
        ) from exc

    snap = Path(snapshot_download(checkpoint, cache_dir=str(cache_dir) if cache_dir else None))
    cfg_file = snap / "config.yaml"
    if not cfg_file.exists():
        return []
    real = cfg_file.resolve()
    text = real.read_text()
    changes: list[str] = []

    if device != "cuda":
        n = text.count("device: cuda")
        if n:
            text = text.replace("device: cuda", f"device: {device}")
            changes.append(f"device: cuda -> {device} ({n} occurrences)")

    # Any non-zero worker count deadlocks on macOS spawn semantics.
    import re

    workers = re.findall(r"num_workers:\s*(\d+)", text)
    if any(int(w) > 0 for w in workers):
        text = re.sub(r"num_workers:\s*\d+", "num_workers: 0", text)
        changes.append("num_workers -> 0 (avoids silent DataLoader deadlock)")

    if changes:
        real.write_text(text)
    return changes


def patch_whisperx_compute_type() -> str | None:
    """Fix TRIBE's hardcoded float16, which CPUs cannot execute.

    Only relevant if you let TRIBE call whisperx itself; we normally bypass it.
    """
    try:
        from .quiet import quiet_imports

        with quiet_imports():
            from tribev2 import eventstransforms
    except ModuleNotFoundError:
        return None
    src = Path(eventstransforms.__file__)
    text = src.read_text()
    needle = '        compute_type = "float16"\n'
    if needle in text:
        text = text.replace(
            needle,
            '        compute_type = "float16" if device == "cuda" else "int8"\n',
        )
        src.write_text(text)
        return "compute_type float16 -> int8 on CPU"
    return None


# -------------------------------------------------------------- the runner


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ModuleNotFoundError:
        pass
    # Apple MPS is deliberately not chosen: TRIBE's encoders fall back to CPU
    # mid-graph and the run hangs rather than erroring.
    return "cpu"


def run(
    video: Path,
    workdir: Path,
    cfg: Config,
    progress: ProgressFn | None = None,
) -> tuple[object, list[str], list[str]]:
    """Score one video. Returns (predictions, modalities, notes).

    predictions is an (timesteps, 20484) numpy array of predicted fMRI response
    on the fsaverage5 cortical surface.
    """

    def tick(stage: str, frac: float, detail: str = "") -> None:
        if progress:
            progress(stage, frac, detail)

    try:
        import numpy as np

        from .quiet import captured_output, quiet_imports, relay_tqdm

        with quiet_imports():
            from tribev2 import TribeModel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TribeError(
            "TRIBE is not installed. Install the optional extras and the model:\n"
            "  pip install 'video-performance-analyzer[tribe]'\n"
            "  pip install git+https://github.com/facebookresearch/tribev2.git"
        ) from exc

    notes: list[str] = []
    raw_log: list[str] = []
    device = resolve_device(cfg.tribe.device)
    tick("preparing", 0.05, f"device={device}")

    # Everything below touches libraries that print at unpredictable points.
    # Divert all of it so the progress display stays readable; the text is kept
    # in raw_log for --verbose and surfaced in full if the run fails.
    stack = contextlib.ExitStack()
    stack.enter_context(quiet_imports())
    stack.enter_context(captured_output(raw_log))
    with stack:
        changes = patch_checkpoint_config(cfg.tribe.checkpoint, cfg.tribe_cache, device)
        notes.extend(changes)
        if fix := patch_whisperx_compute_type():
            notes.append(fix)

        tick("preparing", 0.12, "normalising video")
        prepared = prepare(video, workdir, cfg, keep_audio=True)

        wav = extract_wav(prepared, workdir / "prepared.wav")
        if wav:
            tick("transcribing", 0.20, "word timings" if cfg.tribe.enable_language else "skipped")
            write_word_timings(wav, cfg.tribe.whisper_model, enable=cfg.tribe.enable_language)
            if not cfg.tribe.enable_language:
                notes.append(
                    "language pathway off (needs gated meta-llama/Llama-3.2-1B); "
                    "scored on vision + audition"
                )

        tick("loading", 0.25, cfg.tribe.checkpoint)
        model = TribeModel.from_pretrained(
            cfg.tribe.checkpoint,
            cache_folder=str(cfg.tribe_cache) if cfg.tribe_cache else None,
            device=device,
        )

        tick("extracting", 0.30, "building events")

        def relay_extract(desc: str, done: int, total: int) -> None:
            if total:
                tick("extracting", 0.30 + 0.08 * (done / total), desc[:34])

        with relay_tqdm(relay_extract):
            events = model.get_events_dataframe(video_path=str(prepared))
        modalities = sorted(str(t) for t in events.type.unique())

        # The encode is the long part — tens of minutes. Relay its own progress so
        # the display keeps moving instead of freezing at a single percentage.
        mods = ", ".join(modalities)
        tick("encoding", 0.40, mods)

        def relay(desc: str, done: int, total: int) -> None:
            if not total:
                return
            frac = 0.40 + 0.55 * (done / total)
            label = (desc or "encoding").lower().replace("encoding video", "encoding")
            tick(label.strip() or "encoding", min(frac, 0.95), f"chunk {done}/{total} · {mods}")

        with relay_tqdm(relay):
            preds, _ = model.predict(events=events, verbose=False)

        preds = np.asarray(preds)
    tick("done", 1.0, f"{preds.shape[0]} timesteps")
    if raw_log:
        # Keep it retrievable without putting it in front of the user by default.
        (workdir / "run.log").write_text(raw_log[0])
        notes.append(f"full library output: {workdir / 'run.log'}")
    return preds, modalities, notes
