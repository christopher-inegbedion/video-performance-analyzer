"""Render evaluations for the terminal, and export them as markdown or HTML."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import explain
from .analysis.compare import Comparison
from .analysis.features import Features

SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], lo: float | None = None, hi: float | None = None) -> str:
    if not values:
        return ""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    rng = (hi - lo) or 1.0
    out = []
    for v in values:
        idx = int(round((v - lo) / rng * (len(SPARK) - 1)))
        out.append(SPARK[max(0, min(idx, len(SPARK) - 1))])
    return "".join(out)


def _verdict_colour(v: str) -> str:
    return {"strong": "green", "ahead": "green", "weak": "red",
            "behind": "red", "average": "yellow", "level": "yellow"}.get(v, "white")


def curve_panel(features: Features, width: int = 64) -> Panel:
    trace = features.trace
    line = sparkline(trace, lo=0.6, hi=1.5)
    ticks = f"0s{' ' * max(0, len(line) - 8)}{features.duration_s:.0f}s"
    body = Text()
    body.append(line + "\n", style="cyan")
    body.append(ticks + "\n\n", style="dim")
    body.append("scale 0.6 ", style="dim")
    body.append("▁▂▃▄▅▆▇█", style="cyan")
    body.append(" 1.5  (1.0 = this video's own average)", style="dim")
    return Panel(body, title="predicted response over time", border_style="dim")


def features_table(features: Features) -> Table:
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    t.add_column("metric", style="dim", width=20)
    t.add_column("value", justify="right", width=10)
    t.add_column("meaning", style="dim", overflow="fold")

    def row(name: str, val: str) -> None:
        t.add_row(name, val, explain.metric_help(name).split(".")[0] + ".")

    row("opening_2s", f"{features.opening_2s:.3f}")
    row("closing_2s", f"{features.closing_2s:.3f}")
    row("peak_value", f"{features.peak_value:.3f}")
    row("trough_value", f"{features.trough_value:.3f}")
    row("swing", f"{features.swing:.3f}")
    row("variability", f"{features.variability:.3f}")
    row("decay", f"{features.decay:+.3f}")
    row("sustained_above", f"{features.sustained_above:.1%}")
    return t


def segments_table(features: Features) -> Table | None:
    if not features.segments:
        return None
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("section", width=20)
    t.add_column("span", justify="right")
    t.add_column("mean", justify="right")
    t.add_column("verdict")
    for s in features.segments:
        t.add_row(
            s.label,
            f"{s.start_s:.1f}-{s.end_s:.1f}s",
            f"{s.mean:.3f}",
            Text(s.verdict, style=_verdict_colour(s.verdict)),
        )
    return t


def comparison_table(cmp: Comparison) -> Table:
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("section", width=16)
    t.add_column("yours", justify="right")
    t.add_column(cmp.reference_label[:18] or "reference", justify="right")
    t.add_column("delta", justify="right")
    t.add_column("")
    for s in cmp.sections:
        t.add_row(
            s.label,
            f"{s.subject:.3f}",
            f"{s.reference:.3f}",
            f"{s.delta:+.3f}",
            Text(s.verdict, style=_verdict_colour(s.verdict)),
        )
    return t


def frames_table(key_frames: list[dict]) -> Table:
    """The moments worth actually looking at, with where to find them."""
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("moment", width=18)
    t.add_column("at", justify="right", width=8)
    t.add_column("value", justify="right", width=7)
    t.add_column("frame", style="dim", overflow="fold")
    for k in key_frames:
        value = float(k.get("value", 1.0))
        style = "green" if value >= 1.10 else ("red" if value <= 0.90 else "")
        t.add_row(
            k.get("label", ""),
            f"{float(k.get('time_s', 0)):.1f}s",
            Text(f"{value:.3f}", style=style),
            Path(k.get("path", "")).name,
        )
    return t


def render(
    console: Console,
    features: Features,
    video_label: str,
    comparison: Comparison | None = None,
    recommendations: str | None = None,
    notes: list[str] | None = None,
    key_frames: list[dict] | None = None,
    contact_sheet_path: str | None = None,
) -> None:
    console.print()
    console.print(Panel(
        Text(video_label, style="bold"),
        subtitle=f"{features.duration_s:.1f}s · {features.n_steps} timesteps · "
                 f"{', '.join(features.modalities) or 'unknown modalities'}",
        border_style="cyan",
    ))
    console.print(curve_panel(features))
    console.print(features_table(features))

    if seg := segments_table(features):
        console.print()
        console.print(Panel(seg, title="sections", border_style="dim"))

    if key_frames:
        console.print()
        console.print(Panel(
            frames_table(key_frames),
            title="key moments — the actual frames",
            border_style="blue",
        ))
        trough = next((k for k in key_frames if k.get("label") == "trough"), None)
        if trough:
            console.print(
                f"  [dim]The sag is at {float(trough['time_s']):.1f}s. "
                f"Open that frame before deciding what to change.[/dim]"
            )
        if contact_sheet_path:
            console.print(f"  [dim]all moments in one image: {contact_sheet_path}[/dim]")

    if comparison:
        console.print()
        shape = ("same arc" if comparison.same_shape else "different arc")
        console.print(Panel(
            Group(
                Text(f"shape correlation {comparison.correlation:+.3f}  ({shape})", style="bold"),
                comparison_table(comparison),
            ),
            title=f"vs {comparison.reference_label}",
            border_style="magenta",
        ))
        for n in comparison.notes:
            console.print(f"  [yellow]![/yellow] {n}")

    if features.tail_spike:
        console.print()
        console.print(
            "  [yellow]![/yellow] Final timestep spikes — almost certainly a hard cut, "
            "not your ending landing."
        )

    for n in notes or []:
        console.print(f"  [dim]·[/dim] [dim]{n}[/dim]")

    if recommendations:
        from rich.markdown import Markdown

        console.print()
        console.print(Panel(Markdown(recommendations), title="recommendations",
                            border_style="green"))


# ------------------------------------------------------------------ export


def to_markdown(
    features: Features,
    video_label: str,
    comparison: Comparison | None = None,
    recommendations: str | None = None,
    evaluation_id: str = "",
    notes: list[str] | None = None,
    key_frames: list[dict] | None = None,
    contact_sheet_path: str | None = None,
    report_dir: Path | None = None,
) -> str:
    L: list[str] = []
    L.append(f"# Video analysis — {video_label}")
    L.append("")
    L.append(f"*Generated {datetime.now():%Y-%m-%d %H:%M} · evaluation `{evaluation_id}`*")
    L.append("")
    L.append(f"- Duration: {features.duration_s:.1f}s ({features.n_steps} timesteps)")
    L.append(f"- Modalities scored: {', '.join(features.modalities) or 'unknown'}")
    L.append("")
    L.append("## Headline metrics")
    L.append("")
    L.append("| metric | value | what it means |")
    L.append("|---|---:|---|")
    for name, val in [
        ("opening_2s", f"{features.opening_2s:.3f}"),
        ("closing_2s", f"{features.closing_2s:.3f}"),
        ("peak_value", f"{features.peak_value:.3f} @ {features.peak_time_s:.1f}s"),
        ("trough_value", f"{features.trough_value:.3f} @ {features.trough_time_s:.1f}s"),
        ("swing", f"{features.swing:.3f}"),
        ("variability", f"{features.variability:.3f}"),
        ("decay", f"{features.decay:+.3f}"),
        ("sustained_above", f"{features.sustained_above:.1%}"),
    ]:
        L.append(f"| `{name}` | {val} | {explain.metric_help(name)} |")
    L.append("")
    L.append(f"Curve: `{sparkline(features.trace, 0.6, 1.5)}`")
    L.append("")

    if features.segments:
        L.append("## Sections")
        L.append("")
        L.append("| section | span | mean | verdict |")
        L.append("|---|---:|---:|---|")
        for s in features.segments:
            L.append(f"| {s.label} | {s.start_s:.1f}-{s.end_s:.1f}s | {s.mean:.3f} | {s.verdict} |")
        L.append("")

    if key_frames:
        L.append("## Key moments")
        L.append("")
        L.append("The frames at the moments the curve flags. A timestamp is not a "
                 "decision; these are what you actually change.")
        L.append("")
        for k in key_frames:
            path = Path(k.get("path", ""))
            # Relative links so the report stays portable next to its frames.
            link = (
                os.path.relpath(path, report_dir)
                if report_dir and path.exists()
                else path.as_posix()
            )
            L.append(f"### {k.get('label', '')} — {float(k.get('time_s', 0)):.1f}s "
                     f"({float(k.get('value', 1)):.3f})")
            L.append("")
            L.append(f"![{k.get('label', '')}]({link})")
            L.append("")
            if note := k.get("note"):
                L.append(f"> {note}")
                L.append("")
        if contact_sheet_path:
            sheet = Path(contact_sheet_path)
            link = (
                os.path.relpath(sheet, report_dir)
                if report_dir and sheet.exists()
                else sheet.as_posix()
            )
            L.append(f"![all key moments]({link})")
            L.append("")

    if comparison:
        L.append(f"## Compared against: {comparison.reference_label}")
        L.append("")
        L.append(f"Shape correlation **{comparison.correlation:+.3f}** "
                 f"({'same arc' if comparison.same_shape else 'different arc'})")
        L.append("")
        L.append("| section | yours | reference | delta | |")
        L.append("|---|---:|---:|---:|---|")
        for s in comparison.sections:
            L.append(f"| {s.label} | {s.subject:.3f} | {s.reference:.3f} | "
                     f"{s.delta:+.3f} | {s.verdict} |")
        L.append("")
        for n in comparison.notes:
            L.append(f"> {n}")
        L.append("")

    if notes:
        L.append("## Run notes")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")

    if recommendations:
        L.append("## Recommendations")
        L.append("")
        L.append(recommendations)
        L.append("")

    L.append("---")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append(explain.WHAT_IT_MEASURES)
    L.append("")
    L.append("### Limits")
    L.append("")
    L.append(explain.WHAT_IT_IS_NOT)
    L.append("")
    L.append("### Known artefacts")
    L.append("")
    L.append(explain.KNOWN_ARTEFACTS)
    return "\n".join(L)


def _localise_frames(
    key_frames: list[dict] | None,
    contact_sheet_path: str | None,
    report_path: Path,
) -> tuple[list[dict] | None, str | None]:
    """Copy frames beside the report so it can be moved or sent as a unit."""
    if not key_frames and not contact_sheet_path:
        return key_frames, contact_sheet_path

    asset_dir = report_path.parent / f"{report_path.stem}_frames"
    asset_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    for frame in key_frames or []:
        source = Path(frame.get("path", ""))
        if not source.exists():
            copied.append(frame)
            continue
        target = asset_dir / source.name
        with contextlib.suppress(OSError):
            shutil.copy2(source, target)
        copied.append({**frame, "path": str(target)})

    sheet = contact_sheet_path
    if contact_sheet_path and Path(contact_sheet_path).exists():
        target = asset_dir / Path(contact_sheet_path).name
        with contextlib.suppress(OSError):
            shutil.copy2(contact_sheet_path, target)
            sheet = str(target)

    return copied, sheet


def export(
    path: Path,
    features: Features,
    video_label: str,
    comparison: Comparison | None = None,
    recommendations: str | None = None,
    evaluation_id: str = "",
    notes: list[str] | None = None,
    key_frames: list[dict] | None = None,
    contact_sheet_path: str | None = None,
) -> Path:
    # Copy the frames next to the report. Without this the links point back into
    # the data directory, so the moment someone moves or sends the report the
    # images break — which defeats the point of embedding them.
    local_frames, local_sheet = _localise_frames(
        key_frames, contact_sheet_path, path
    )
    md = to_markdown(
        features, video_label, comparison, recommendations, evaluation_id, notes,
        local_frames, local_sheet, report_dir=path.parent,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "video": video_label,
            "features": features.to_dict(),
            "comparison": comparison.to_dict() if comparison else None,
            "recommendations": recommendations,
            "notes": notes or [],
            "key_frames": key_frames or [],
            "contact_sheet": contact_sheet_path,
        }
        path.write_text(json.dumps(payload, indent=2))
    else:
        path.write_text(md)
    return path
