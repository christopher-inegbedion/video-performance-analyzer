"""Command line interface for video-performance-analyzer."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from . import config as config_mod
from . import db, explain, learn, metrics, pipeline, recommend, report
from .providers import LLMError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Predict how a video lands, compare it to a reference and your own history, "
         "and get recommendations that learn from real performance.",
)
list_app = typer.Typer(no_args_is_help=True, help="List stored videos, evaluations, reports.")
metrics_app = typer.Typer(no_args_is_help=True, help="Record real-world performance.")
config_app = typer.Typer(no_args_is_help=True, help="Inspect and edit configuration.")
app.add_typer(list_app, name="list")
app.add_typer(metrics_app, name="metrics")
app.add_typer(config_app, name="config")

console = Console()


def _fail(msg: str, hint: str = "") -> None:
    console.print(f"[red]✗[/red] {msg}")
    if hint:
        console.print(f"  [dim]{hint}[/dim]")
    raise typer.Exit(1)


def _ts(value) -> str:
    if not value:
        return "—"
    return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")


def _ensure_db() -> None:
    db.init()


# ---------------------------------------------------------------- analyse


@app.command()
def analyse(
    video: Path = typer.Argument(..., help="The video you made."),
    reference: Path = typer.Option(
        None, "--reference", "-r",
        help="A video to compare against — a competitor's ad, your last post, "
             "something you want to emulate."),
    label: str = typer.Option(None, "--label", "-l", help="Name for this video."),
    reference_label: str = typer.Option(None, "--reference-label"),
    segments: str = typer.Option(
        None, "--segments", "-s",
        help='Your real structure, e.g. "hook:0-3,montage:3-16,card:16-19,end:19-24"'),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the recommendation step."),
    export_to: Path = typer.Option(None, "--export", "-o", help="Write a .md or .json report."),
    ask: str = typer.Option(None, "--ask", help="An extra instruction for the analyst."),
) -> None:
    """Score a video and explain what the numbers mean."""
    _ensure_db()
    cfg = config_mod.load()

    if not video.exists():
        _fail(f"No such file: {video}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("starting", total=1.0, detail="")

        def tick(stage: str, frac: float, detail: str = "") -> None:
            prog.update(task, description=stage, completed=max(frac, 0.01), detail=detail)

        try:
            result = pipeline.evaluate(
                cfg, video, reference, label, reference_label,
                segments_spec=segments, progress=tick,
            )
        except Exception as exc:  # noqa: BLE001
            prog.stop()
            _fail(str(exc), "Run `vpa doctor` to check your setup.")

    recs = None
    if not no_llm:
        try:
            with console.status("[dim]asking the model for recommendations…[/dim]"):
                recs, evidence = recommend.generate(
                    cfg, result.features, result.video_label,
                    result.comparison, result.history, result.learned,
                    extra_instruction=ask or "",
                )
            db.add_recommendation(
                result.evaluation_id, recs, evidence["model"], "initial", evidence
            )
        except LLMError as exc:
            console.print(f"[yellow]![/yellow] Recommendations unavailable: {exc}")
            console.print("[dim]  The analysis above is unaffected. "
                          "Re-run later with `vpa recommend <id>`.[/dim]")

    report.render(console, result.features, result.video_label,
                  result.comparison, recs, result.notes)

    if result.history and result.history.notes:
        for n in result.history.notes:
            console.print(f"  [dim]·[/dim] [dim]{n}[/dim]")

    console.print()
    short = result.evaluation_id[:12]
    console.print(
        f"  evaluation [cyan]{result.evaluation_id}[/cyan]  "
        f"[dim]· vpa show {short} · vpa ask {short}[/dim]"
    )

    if export_to:
        path = report.export(export_to, result.features, result.video_label,
                             result.comparison, recs, result.evaluation_id, result.notes)
        console.print(f"  report written to [green]{path}[/green]")


# ------------------------------------------------------------------ show


@app.command()
def show(
    evaluation: str = typer.Argument(..., help="Evaluation id (a prefix is fine)."),
    export_to: Path = typer.Option(None, "--export", "-o"),
    generation: int = typer.Option(
        None, "--generation", "-g", help="Which recommendation generation to show."),
) -> None:
    """Show a past evaluation and every recommendation written for it."""
    _ensure_db()
    try:
        row = db.resolve_evaluation(evaluation)
    except ValueError as exc:
        _fail(str(exc))
    if not row:
        _fail(f"No evaluation matching '{evaluation}'.", "Try `vpa list evals`.")

    features, comparison, _ = pipeline.load_result(row["id"])
    if not features:
        status = row["status"]
        if status == "failed":
            _fail(f"That evaluation failed: {row['error']}")
        _fail(f"That evaluation is '{status}', no results stored yet.")

    vrow = db.get_video(row["video_id"])
    label = vrow["label"] if vrow else row["id"]

    recs = db.recommendations_for(row["id"])
    chosen = None
    if recs:
        chosen = recs[-1] if generation is None else next(
            (r for r in recs if r["generation"] == generation), None)

    report.render(console, features, label, comparison,
                  chosen["body"] if chosen else None)

    if len(recs) > 1:
        console.print()
        console.print("  [dim]recommendation generations:[/dim]")
        for r in recs:
            marker = "→" if chosen and r["id"] == chosen["id"] else " "
            console.print(
                f"  {marker} gen {r['generation']}  {r['kind']:<11} "
                f"{_ts(r['created_at'])}  [dim]{r['model']}[/dim]"
            )
        console.print(f"  [dim]see an older one: vpa show {row['id'][:12]} -g 1[/dim]")

    if export_to:
        path = report.export(export_to, features, label, comparison,
                             chosen["body"] if chosen else None, row["id"])
        console.print(f"\n  report written to [green]{path}[/green]")


# ------------------------------------------------------------------ list


@list_app.command("evals")
def list_evals(
    limit: int = typer.Option(20, "--limit", "-n"),
    status: str = typer.Option(None, "--status"),
) -> None:
    """Every evaluation, newest first."""
    _ensure_db()
    rows = db.list_evaluations(limit=limit, status=status)
    if not rows:
        console.print("[dim]No evaluations yet. Run `vpa analyse <video>`.[/dim]")
        return
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("id", style="cyan")
    t.add_column("video")
    t.add_column("reference", style="dim")
    t.add_column("status")
    t.add_column("when", style="dim")
    for r in rows:
        colour = {"done": "green", "failed": "red"}.get(r["status"], "yellow")
        pct = "" if r["status"] in {"done", "failed"} else f" {r['progress']:.0%}"
        t.add_row(
            r["id"][:14],
            (r["video_label"] or "")[:26],
            (r["reference_label"] or "—")[:20],
            Text(f"{r['status']}{pct}", style=colour),
            _ts(r["created_at"]),
        )
    console.print(t)


@list_app.command("videos")
def list_videos_cmd(kind: str = typer.Option(None, "--kind", help="subject | reference")) -> None:
    """Every registered video, and whether it has performance data."""
    _ensure_db()
    rows = db.list_videos(kind=kind)
    if not rows:
        console.print("[dim]No videos registered yet.[/dim]")
        return
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("id", style="cyan")
    t.add_column("label")
    t.add_column("kind", style="dim")
    t.add_column("dur", justify="right")
    t.add_column("metrics", justify="right")
    for r in rows:
        perf = db.performance_for(r["id"])
        t.add_row(
            r["id"][:14],
            (r["label"] or "")[:30],
            r["kind"],
            f"{(r['duration_s'] or 0):.0f}s",
            Text(str(len(perf)), style="green" if perf else "dim"),
        )
    console.print(t)


@list_app.command("reports")
def list_reports() -> None:
    """Recommendations written, across all evaluations."""
    _ensure_db()
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT r.*, v.label FROM recommendations r
               JOIN evaluations e ON e.id = r.evaluation_id
               JOIN videos v ON v.id = e.video_id
               ORDER BY r.created_at DESC LIMIT 50"""
        ).fetchall()
    if not rows:
        console.print("[dim]No recommendations generated yet.[/dim]")
        return
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("evaluation", style="cyan")
    t.add_column("video")
    t.add_column("gen", justify="right")
    t.add_column("kind")
    t.add_column("when", style="dim")
    for r in rows:
        t.add_row(
            r["evaluation_id"][:14], (r["label"] or "")[:26], str(r["generation"]),
            Text(r["kind"], style="magenta" if r["kind"] == "retroactive" else "white"),
            _ts(r["created_at"]),
        )
    console.print(t)


# --------------------------------------------------------------- metrics


@metrics_app.command("add")
def metrics_add(
    video: str = typer.Argument(..., help="Video label or id."),
    views: int = typer.Option(None, "--views"),
    likes: int = typer.Option(None, "--likes"),
    comments: int = typer.Option(None, "--comments"),
    shares: int = typer.Option(None, "--shares"),
    saves: int = typer.Option(None, "--saves"),
    watch_through: float = typer.Option(None, "--watch-through", help="0-1 or a percentage."),
    followers: int = typer.Option(None, "--followers"),
    platform: str = typer.Option(None, "--platform"),
    posted_at: str = typer.Option(None, "--posted-at", help="YYYY-MM-DD"),
    notes: str = typer.Option(None, "--notes"),
) -> None:
    """Record what a published video actually did."""
    _ensure_db()
    try:
        metrics.record(
            video, platform, views, likes, comments, shares, saves,
            watch_through, followers, posted_at, notes,
        )
    except ValueError as exc:
        _fail(str(exc))
    console.print(f"[green]✓[/green] recorded performance for [cyan]{video}[/cyan]")
    model = learn.build()
    console.print(f"  [dim]{model.summary_text().splitlines()[0]}[/dim]")
    if stale := learn.stale_evaluations():
        console.print(
            f"  [yellow]![/yellow] {len(stale)} past evaluation(s) now have advice that "
            "predates this data."
        )
        console.print("  [dim]  refresh them with: vpa retrofit[/dim]")


@metrics_app.command("import")
def metrics_import(csv_path: Path = typer.Argument(..., help="CSV file to import.")) -> None:
    """Bulk-import performance from a CSV."""
    _ensure_db()
    try:
        n, problems = metrics.import_csv(csv_path)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(f"[green]✓[/green] imported {n} row(s)")
    for p in problems:
        console.print(f"  [yellow]![/yellow] {p}")


@metrics_app.command("template")
def metrics_template(
    target: Path = typer.Option(Path("vpa-metrics.csv"), "--out", "-o")
) -> None:
    """Write a CSV template you can fill in."""
    path = metrics.write_template(target)
    console.print(f"[green]✓[/green] template written to [green]{path}[/green]")


@metrics_app.command("show")
def metrics_show() -> None:
    """What the tool has learned from real performance so far."""
    _ensure_db()
    model = learn.build()
    if not model.n_labelled:
        console.print(Panel(
            "No published performance recorded yet.\n\n"
            "Recommendations currently rest on the model's predictions alone. Add real\n"
            "outcomes and they start being weighted against what actually happened:\n\n"
            "  vpa metrics add my-video --views 12400 --likes 380",
            title="learning", border_style="yellow"))
        return
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("feature")
    t.add_column("r", justify="right")
    t.add_column("n", justify="right")
    t.add_column("reads as")
    t.add_column("confidence")
    for c in model.correlations:
        colour = {"moderate": "green", "weak": "yellow"}.get(c.confidence, "dim")
        t.add_row(c.feature, f"{c.r:+.2f}", str(c.n), c.direction,
                  Text(c.confidence, style=colour))
    console.print(Panel(t, title=f"learned from {model.n_labelled} video(s)",
                        border_style="green" if model.usable else "yellow"))
    console.print(f"[dim]{model.caveat}[/dim]")


# ------------------------------------------------------- recommend/retrofit


def recommend_cmd(
    evaluation: str = typer.Argument(..., help="Evaluation id (prefix ok)."),
    ask: str = typer.Option(None, "--ask", help="Extra instruction for the analyst."),
) -> None:
    """Generate a fresh recommendation for an existing evaluation."""
    _ensure_db()
    cfg = config_mod.load()
    try:
        row = db.resolve_evaluation(evaluation)
    except ValueError as exc:
        _fail(str(exc))
    if not row:
        _fail(f"No evaluation matching '{evaluation}'.")
    features, comparison, _ = pipeline.load_result(row["id"])
    if not features:
        _fail("That evaluation has no stored results.")
    vrow = db.get_video(row["video_id"])
    try:
        with console.status("[dim]thinking…[/dim]"):
            body, evidence = recommend.generate(
                cfg, features, vrow["label"], comparison, None, learn.build(),
                extra_instruction=ask or "",
            )
    except LLMError as exc:
        _fail(str(exc))
    db.add_recommendation(row["id"], body, evidence["model"], "initial", evidence)
    console.print(Panel(Markdown(body), title="recommendations", border_style="green"))


app.command(name="recommend")(recommend_cmd)


@app.command()
def retrofit(
    evaluation: str = typer.Argument(None, help="One evaluation, or omit for all stale ones."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before running."),
) -> None:
    """Revisit old evaluations now that new performance data exists.

    This is the self-improving loop: advice written before you knew how a video
    performed gets a second pass once you do.
    """
    _ensure_db()
    cfg = config_mod.load()
    model = learn.build()

    if evaluation:
        try:
            row = db.resolve_evaluation(evaluation)
        except ValueError as exc:
            _fail(str(exc))
        if not row:
            _fail(f"No evaluation matching '{evaluation}'.")
        targets = [{"evaluation_id": row["id"], "label": None}]
    else:
        targets = learn.stale_evaluations()
        if not targets:
            console.print(
                "[dim]Nothing to revisit — no evaluation has advice older than your "
                "most recent performance data.[/dim]")
            return
        console.print(f"{len(targets)} evaluation(s) have advice predating your latest metrics:")
        for t in targets:
            console.print(f"  [cyan]{t['evaluation_id'][:14]}[/cyan]  {t['label'] or ''}")
        if not yes and not typer.confirm("Rewrite recommendations for these?"):
            raise typer.Abort()

    for t in targets:
        eid = t["evaluation_id"]
        features, comparison, _ = pipeline.load_result(eid)
        if not features:
            continue
        erow = db.get_evaluation(eid)
        vrow = db.get_video(erow["video_id"])
        prev = db.recommendations_for(eid)
        previous_body = prev[-1]["body"] if prev else "(no previous recommendation)"
        perf = [dict(p) for p in db.performance_for(erow["video_id"])]
        try:
            with console.status(f"[dim]revisiting {eid[:12]}…[/dim]"):
                body, evidence = recommend.generate_retroactive(
                    cfg, features, vrow["label"], previous_body, model, perf
                )
        except LLMError as exc:
            console.print(f"[yellow]![/yellow] {eid[:12]}: {exc}")
            continue
        db.add_recommendation(eid, body, evidence["model"], "retroactive", evidence)
        console.print(Panel(Markdown(body),
                            title=f"revised · {vrow['label']}", border_style="magenta"))


# -------------------------------------------------------------------- ask


@app.command()
def ask(
    evaluation: str = typer.Argument(None, help="Ground the conversation in one evaluation."),
    question: str = typer.Option(None, "--q", help="Ask once and exit."),
) -> None:
    """Ask questions about an evaluation and get grounded answers."""
    _ensure_db()
    cfg = config_mod.load()
    features = comparison = None
    label = "your videos"
    eid = None

    if evaluation:
        try:
            row = db.resolve_evaluation(evaluation)
        except ValueError as exc:
            _fail(str(exc))
        if not row:
            _fail(f"No evaluation matching '{evaluation}'.")
        eid = row["id"]
        features, comparison, _ = pipeline.load_result(eid)
        vrow = db.get_video(row["video_id"])
        label = vrow["label"]

    model = learn.build()

    def one(q: str) -> None:
        db.add_chat(eid, "user", q)
        try:
            chunks = recommend.answer_question(
                cfg, q, features, label, comparison, model,
                db.chat_history(eid, limit=20)[:-1], stream=True,
            )
            console.print()
            acc: list[str] = []
            for piece in chunks:
                console.print(piece, end="")
                acc.append(piece)
            console.print("\n")
            db.add_chat(eid, "assistant", "".join(acc))
        except LLMError as exc:
            _fail(str(exc))

    if question:
        one(question)
        return

    console.print(Panel(
        f"Asking about [cyan]{label}[/cyan].\n"
        "[dim]Type a question, or /exit to leave.[/dim]",
        border_style="cyan"))
    while True:
        try:
            q = console.input("[bold cyan]? [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not q:
            continue
        if q in {"/exit", "/quit", "exit", "quit"}:
            break
        one(q)


# ----------------------------------------------------------------- explain


def explain_cmd(
    topic: str = typer.Argument(None, help="what | measures | limits | artefacts | lag | "
                                           "modalities | licence — or a metric name."),
) -> None:
    """Explain TRIBE, the metrics, and what none of it can tell you."""
    if not topic:
        console.print(Markdown(explain.full_text()))
        return
    key = topic.lower()
    if key in explain.SECTIONS:
        title, body = explain.SECTIONS[key]
        console.print(Panel(body, title=title, border_style="cyan"))
        return
    if key in explain.METRIC_GLOSSARY:
        console.print(Panel(explain.METRIC_GLOSSARY[key], title=key, border_style="cyan"))
        return
    _fail(f"Nothing to explain for '{topic}'.",
          "Try: " + ", ".join(list(explain.SECTIONS) + list(explain.METRIC_GLOSSARY)[:4]))


app.command(name="explain")(explain_cmd)


# ------------------------------------------------------------------ config


@config_app.command("init")
def config_init(force: bool = typer.Option(False, "--force")) -> None:
    """Write a commented config file you can edit."""
    path = config_mod.write_default(force=force)
    console.print(f"[green]✓[/green] config at [green]{path}[/green]")


@config_app.command("show")
def config_show() -> None:
    """Show effective configuration and where it came from."""
    cfg = config_mod.load()
    t = Table(show_header=True, header_style="bold", box=None)
    t.add_column("setting", style="dim")
    t.add_column("value")
    t.add_row("llm.provider", cfg.llm.provider)
    t.add_row("llm.model", cfg.llm.model)
    t.add_row("llm.base_url", cfg.llm.base_url)
    key = cfg.api_key()
    t.add_row(
        cfg.llm.api_key_env,
        Text("set", style="green") if key else Text("NOT SET", style="red"),
    )
    t.add_row("tribe.checkpoint", cfg.tribe.checkpoint)
    t.add_row("tribe.device", cfg.tribe.device)
    t.add_row("tribe.target_fps", str(cfg.tribe.target_fps))
    t.add_row("tribe.enable_language", str(cfg.tribe.enable_language))
    t.add_row("config file", str(config_mod.config_path()))
    t.add_row("data dir", str(config_mod.data_dir()))
    console.print(t)


# ------------------------------------------------------------------ doctor


@app.command()
def doctor() -> None:
    """Check the environment and say exactly what is missing."""
    cfg = config_mod.load()
    ok = True

    def check(name: str, good: bool, detail: str = "", fix: str = "") -> None:
        nonlocal ok
        mark = "[green]✓[/green]" if good else "[red]✗[/red]"
        console.print(f"  {mark} {name}" + (f"  [dim]{detail}[/dim]" if detail else ""))
        if not good:
            ok = False
            if fix:
                # escape() so pip extras like [tribe] aren't eaten as Rich markup
                console.print(f"      [yellow]{escape(fix)}[/yellow]")

    console.print("\n[bold]environment[/bold]")
    import shutil as _sh

    check("ffmpeg", _sh.which("ffmpeg") is not None, fix="brew install ffmpeg")
    check("ffprobe", _sh.which("ffprobe") is not None, fix="brew install ffmpeg")
    check("python", True, f"{sys.version.split()[0]}")

    console.print("\n[bold]models[/bold]")
    try:
        import torch

        check("torch", True, torch.__version__)
        if not (2, 5) <= tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 7):
            console.print("      [yellow]TRIBE wants torch >=2.5,<2.7 — other versions "
                          "may fail at import[/yellow]")
    except ModuleNotFoundError:
        check("torch", False, fix="pip install 'video-performance-analyzer[tribe]'")
    try:
        from .quiet import quiet_imports

        with quiet_imports():
            import tribev2  # noqa: F401

        check("tribev2", True)
    except ModuleNotFoundError:
        check("tribev2", False,
              fix="pip install git+https://github.com/facebookresearch/tribev2.git")
    try:
        import faster_whisper  # noqa: F401

        check("faster-whisper", True, "needed only for the language pathway")
    except ModuleNotFoundError:
        check("faster-whisper", False, "language pathway unavailable",
              "pip install faster-whisper")

    console.print("\n[bold]llm[/bold]")
    check(f"{cfg.llm.api_key_env}", cfg.api_key() is not None,
          cfg.llm.model, f"export {cfg.llm.api_key_env}=...")

    console.print("\n[bold]storage[/bold]")
    _ensure_db()
    check("database", db.db_path().exists(), str(db.db_path()))

    console.print()
    if ok:
        console.print("[green]Everything needed is present.[/green]\n")
    else:
        console.print("[yellow]Some pieces are missing — see the fixes above.[/yellow]\n")


# -------------------------------------------------------------------- tui


@app.command()
def tui() -> None:
    """Interactive terminal session."""
    from .tui import run_tui

    _ensure_db()
    run_tui(console)


@app.callback()
def main() -> None:
    """video-performance-analyzer."""


if __name__ == "__main__":
    app()
