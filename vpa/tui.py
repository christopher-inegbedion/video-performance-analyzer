"""Interactive terminal session.

A plain REPL rather than a full-screen app on purpose: it composes with scroll,
copy-paste and tmux, and it degrades gracefully over ssh. Every action is also
available as a one-shot subcommand, so nothing is locked behind the TUI.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config as config_mod
from . import db, explain, learn, metrics, pipeline, recommend, report
from .providers import LLMError

BANNER = """\
[bold]video-performance-analyzer[/bold]
[dim]predicted neural response · reference comparison · recommendations that learn[/dim]"""

HELP = """\
[bold]analysis[/bold]
  analyse <video> [--ref <video>]   score a video, optionally against a reference
  show <id>                         a past evaluation, with its recommendations
  ask [<id>]                        ask questions, grounded in an evaluation

[bold]history[/bold]
  evals                             every evaluation
  videos                            every registered video
  reports                           every recommendation written

[bold]learning[/bold]
  metrics <video> views=N likes=N   record what a published video did
  learned                           what real performance has taught the tool
  retrofit                          revisit old advice using newer outcomes

[bold]reference[/bold]
  explain [topic]                   what TRIBE measures, and what it cannot
  config                            effective settings
  help · exit\
"""


def _ts(v) -> str:
    return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M") if v else "—"


def _parse_kv(tokens: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in tokens:
        if "=" in t:
            k, _, v = t.partition("=")
            out[k.strip().lower()] = v.strip()
    return out


def run_tui(console: Console) -> None:
    cfg = config_mod.load()
    console.print()
    console.print(Panel(BANNER, border_style="cyan"))

    if not cfg.api_key():
        console.print(
            f"  [yellow]![/yellow] {cfg.llm.api_key_env} is not set — scoring works, "
            "recommendations will not."
        )
    model = learn.build()
    console.print(f"  [dim]{model.summary_text().splitlines()[0]}[/dim]")
    console.print("  [dim]type 'help' for commands[/dim]\n")

    current: str | None = None  # evaluation in focus

    while True:
        prompt = "[bold cyan]vpa[/bold cyan]" + (
            f" [dim]({current[:10]})[/dim]" if current else "") + "[bold cyan] ›[/bold cyan] "
        try:
            raw = console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not raw:
            continue

        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in {"exit", "quit", "/exit", "q"}:
            console.print("[dim]bye[/dim]")
            return

        if cmd in {"help", "?", "/help"}:
            console.print(Panel(HELP, border_style="dim"))
            continue

        # ----------------------------------------------------------- analyse
        if cmd in {"analyse", "analyze"}:
            if not args:
                console.print("  [red]usage:[/red] analyse <video> [--ref <video>] [--label X]")
                continue
            video = Path(args[0]).expanduser()
            ref = None
            label = None
            if "--ref" in args:
                i = args.index("--ref")
                if i + 1 < len(args):
                    ref = Path(args[i + 1]).expanduser()
            if "--label" in args:
                i = args.index("--label")
                if i + 1 < len(args):
                    label = args[i + 1]
            if not video.exists():
                console.print(f"  [red]✗[/red] no such file: {video}")
                continue
            try:
                with console.status("[dim]scoring — this can take several minutes[/dim]") as st:
                    def tick(stage: str, frac: float, detail: str = "") -> None:
                        st.update(f"[dim]{stage} {frac:.0%} {detail}[/dim]")

                    result = pipeline.evaluate(cfg, video, ref, label, progress=tick)
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]✗[/red] {exc}")
                continue

            recs = None
            if cfg.api_key():
                try:
                    with console.status("[dim]asking the model…[/dim]"):
                        recs, evidence = recommend.generate(
                            cfg, result.features, result.video_label,
                            result.comparison, result.history, result.learned)
                    db.add_recommendation(result.evaluation_id, recs,
                                          evidence["model"], "initial", evidence)
                except LLMError as exc:
                    console.print(f"  [yellow]![/yellow] {exc}")

            report.render(console, result.features, result.video_label,
                          result.comparison, recs, result.notes)
            current = result.evaluation_id
            console.print(f"\n  [dim]in focus: {current}[/dim]")
            continue

        # -------------------------------------------------------------- show
        if cmd == "show":
            target = args[0] if args else current
            if not target:
                console.print("  [red]usage:[/red] show <evaluation-id>")
                continue
            try:
                row = db.resolve_evaluation(target)
            except ValueError as exc:
                console.print(f"  [red]✗[/red] {exc}")
                continue
            if not row:
                console.print(f"  [red]✗[/red] no evaluation matching '{target}'")
                continue
            features, comparison, _ = pipeline.load_result(row["id"])
            if not features:
                console.print(f"  [yellow]![/yellow] status is '{row['status']}'"
                              + (f": {row['error']}" if row["error"] else ""))
                continue
            vrow = db.get_video(row["video_id"])
            recs = db.recommendations_for(row["id"])
            report.render(console, features, vrow["label"], comparison,
                          recs[-1]["body"] if recs else None)
            current = row["id"]
            continue

        # ------------------------------------------------------------- lists
        if cmd == "evals":
            rows = db.list_evaluations(limit=20)
            if not rows:
                console.print("  [dim]none yet[/dim]")
                continue
            t = Table(box=None, header_style="bold")
            t.add_column("id", style="cyan")
            t.add_column("video")
            t.add_column("status")
            t.add_column("when", style="dim")
            for r in rows:
                colour = {"done": "green", "failed": "red"}.get(r["status"], "yellow")
                t.add_row(r["id"][:14], (r["video_label"] or "")[:28],
                          Text(r["status"], style=colour), _ts(r["created_at"]))
            console.print(t)
            continue

        if cmd == "videos":
            rows = db.list_videos()
            if not rows:
                console.print("  [dim]none yet[/dim]")
                continue
            t = Table(box=None, header_style="bold")
            t.add_column("id", style="cyan")
            t.add_column("label")
            t.add_column("kind", style="dim")
            t.add_column("metrics", justify="right")
            for r in rows:
                n = len(db.performance_for(r["id"]))
                t.add_row(r["id"][:14], (r["label"] or "")[:30], r["kind"],
                          Text(str(n), style="green" if n else "dim"))
            console.print(t)
            continue

        if cmd == "reports":
            with db.connect() as conn:
                rows = conn.execute(
                    """SELECT r.*, v.label FROM recommendations r
                       JOIN evaluations e ON e.id=r.evaluation_id
                       JOIN videos v ON v.id=e.video_id
                       ORDER BY r.created_at DESC LIMIT 25"""
                ).fetchall()
            if not rows:
                console.print("  [dim]none yet[/dim]")
                continue
            t = Table(box=None, header_style="bold")
            t.add_column("evaluation", style="cyan")
            t.add_column("video")
            t.add_column("gen", justify="right")
            t.add_column("kind")
            t.add_column("when", style="dim")
            for r in rows:
                t.add_row(r["evaluation_id"][:14], (r["label"] or "")[:24],
                          str(r["generation"]),
                          Text(r["kind"],
                               style="magenta" if r["kind"] == "retroactive" else "white"),
                          _ts(r["created_at"]))
            console.print(t)
            continue

        # ----------------------------------------------------------- metrics
        if cmd == "metrics":
            if not args:
                console.print("  [red]usage:[/red] metrics <video> views=12400 likes=380")
                continue
            kv = _parse_kv(args[1:])
            try:
                metrics.record(
                    args[0],
                    platform=kv.get("platform"),
                    views=int(kv["views"]) if "views" in kv else None,
                    likes=int(kv["likes"]) if "likes" in kv else None,
                    comments=int(kv["comments"]) if "comments" in kv else None,
                    shares=int(kv["shares"]) if "shares" in kv else None,
                    saves=int(kv["saves"]) if "saves" in kv else None,
                    posted_at=kv.get("posted_at"),
                )
            except (ValueError, KeyError) as exc:
                console.print(f"  [red]✗[/red] {exc}")
                continue
            console.print("  [green]✓[/green] recorded")
            if stale := learn.stale_evaluations():
                console.print(f"  [yellow]![/yellow] {len(stale)} evaluation(s) now have "
                              "stale advice — run 'retrofit'")
            continue

        if cmd == "learned":
            m = learn.build()
            if not m.n_labelled:
                console.print("  [dim]nothing learned yet — record some metrics first[/dim]")
                continue
            t = Table(box=None, header_style="bold")
            t.add_column("feature")
            t.add_column("r", justify="right")
            t.add_column("reads as")
            t.add_column("confidence")
            for c in m.correlations:
                colour = {"moderate": "green", "weak": "yellow"}.get(c.confidence, "dim")
                t.add_row(c.feature, f"{c.r:+.2f}", c.direction,
                          Text(c.confidence, style=colour))
            console.print(Panel(t, title=f"learned from {m.n_labelled} video(s)",
                                border_style="green" if m.usable else "yellow"))
            console.print(f"  [dim]{m.caveat}[/dim]")
            continue

        if cmd == "retrofit":
            targets = learn.stale_evaluations()
            if not targets:
                console.print("  [dim]nothing to revisit[/dim]")
                continue
            m = learn.build()
            for t in targets:
                features, comparison, _ = pipeline.load_result(t["evaluation_id"])
                if not features:
                    continue
                erow = db.get_evaluation(t["evaluation_id"])
                vrow = db.get_video(erow["video_id"])
                prev = db.recommendations_for(t["evaluation_id"])
                try:
                    with console.status(f"[dim]revisiting {t['evaluation_id'][:10]}…[/dim]"):
                        body, ev = recommend.generate_retroactive(
                            cfg, features, vrow["label"],
                            prev[-1]["body"] if prev else "(none)", m,
                            [dict(p) for p in db.performance_for(erow["video_id"])])
                except LLMError as exc:
                    console.print(f"  [yellow]![/yellow] {exc}")
                    continue
                db.add_recommendation(t["evaluation_id"], body, ev["model"], "retroactive", ev)
                console.print(Panel(Markdown(body), title=f"revised · {vrow['label']}",
                                    border_style="magenta"))
            continue

        # --------------------------------------------------------------- ask
        if cmd == "ask":
            target = args[0] if args and not args[0].startswith("-") else current
            question = " ".join(args[1:] if target in args else args).strip()
            features = comparison = None
            label = "your videos"
            eid = None
            if target:
                try:
                    row = db.resolve_evaluation(target)
                except ValueError as exc:
                    console.print(f"  [red]✗[/red] {exc}")
                    continue
                if row:
                    eid = row["id"]
                    features, comparison, _ = pipeline.load_result(eid)
                    label = db.get_video(row["video_id"])["label"]
            if not question:
                console.print(f"  [dim]asking about {label} — blank line to stop[/dim]")
            m = learn.build()
            while True:
                q = question or console.input("  [bold cyan]? [/bold cyan]").strip()
                question = ""
                if not q:
                    break
                db.add_chat(eid, "user", q)
                try:
                    acc: list[str] = []
                    console.print()
                    for piece in recommend.answer_question(
                        cfg, q, features, label, comparison, m,
                        db.chat_history(eid, limit=20)[:-1], stream=True
                    ):
                        console.print(piece, end="")
                        acc.append(piece)
                    console.print("\n")
                    db.add_chat(eid, "assistant", "".join(acc))
                except LLMError as exc:
                    console.print(f"  [red]✗[/red] {exc}")
                    break
            continue

        # ----------------------------------------------------------- explain
        if cmd == "explain":
            topic = args[0].lower() if args else None
            if not topic:
                console.print(Markdown(explain.full_text()))
            elif topic in explain.SECTIONS:
                title, body = explain.SECTIONS[topic]
                console.print(Panel(body, title=title, border_style="cyan"))
            elif topic in explain.METRIC_GLOSSARY:
                console.print(Panel(explain.METRIC_GLOSSARY[topic], title=topic,
                                    border_style="cyan"))
            else:
                console.print(f"  [dim]nothing for '{topic}'. try: "
                              f"{', '.join(explain.SECTIONS)}[/dim]")
            continue

        if cmd == "config":
            t = Table(box=None, header_style="bold")
            t.add_column("setting", style="dim")
            t.add_column("value")
            t.add_row("model", cfg.llm.model)
            t.add_row("provider", f"{cfg.llm.provider} ({cfg.llm.base_url})")
            t.add_row("api key", Text("set", style="green") if cfg.api_key()
                      else Text("NOT SET", style="red"))
            t.add_row("checkpoint", cfg.tribe.checkpoint)
            t.add_row("device", cfg.tribe.device)
            t.add_row("language pathway", str(cfg.tribe.enable_language))
            t.add_row("config file", str(config_mod.config_path()))
            console.print(t)
            continue

        console.print(f"  [dim]unknown command '{cmd}' — type 'help'[/dim]")
