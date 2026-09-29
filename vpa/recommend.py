"""Build LLM prompts from evaluation data and produce actionable recommendations.

Design rule: the model is given evidence and constraints, never raw permission to
speculate. Every caveat from `explain.py` is injected, so the recommendations
inherit the same honesty about what TRIBE can and cannot tell you.
"""

from __future__ import annotations

from typing import Any

from . import explain, learn
from .analysis.compare import Comparison, HistoryPosition
from .analysis.features import Features
from .config import Config
from .providers import Message, get_provider

SYSTEM_PROMPT = """\
You are a video editing analyst. You advise on short-form video edits using
predicted neural-response data from Meta's TRIBE v2 brain-encoding model.

You must obey these rules:

1. Recommendations must be CONCRETE and about the EDIT: what to cut, shorten,
   move, or hold longer, with timecodes. Never give generic marketing advice.
2. Distinguish clearly between what the data shows and what you are inferring.
   Say "the data shows" and "I suspect" and never blur them.
3. Never claim TRIBE predicts attention, watch time, virality or sales. It
   predicts fMRI response. If you speculate about audience behaviour, label it.
4. Treat a spike on the final timestep as a cut artefact, not a finding.
5. If the evidence is thin, say so plainly and give fewer recommendations. Three
   well-grounded notes beat ten speculative ones.
6. Where real published performance data exists, weight it ABOVE the model's
   predictions. Actual outcomes beat predicted ones.
7. Be concise. Use short paragraphs and bullet lists. No preamble, no flattery.

Output format: markdown. Start with a one-paragraph verdict, then a section
"What the data shows", then "What I would change", then "What I am unsure about".\
"""


def _trace_table(features: Features, max_rows: int = 40) -> str:
    """Render the curve compactly enough to fit in a prompt."""
    step = features.step_s
    trace = features.trace
    if len(trace) > max_rows:
        stride = len(trace) // max_rows + 1
        rows = [(i * step, v) for i, v in enumerate(trace) if i % stride == 0]
    else:
        rows = [(i * step, v) for i, v in enumerate(trace)]
    return "\n".join(f"  {t:6.1f}s  {v:.3f}" for t, v in rows)


def build_context(
    features: Features,
    video_label: str,
    comparison: Comparison | None = None,
    history: HistoryPosition | None = None,
    learned: learn.LearnedModel | None = None,
    segments_note: str = "",
) -> str:
    parts: list[str] = []
    parts.append(f"# Video under analysis: {video_label}")
    parts.append(
        f"Duration {features.duration_s:.1f}s · {features.n_steps} timesteps "
        f"(~{features.step_s:.2f}s each) · modalities scored: "
        f"{', '.join(features.modalities) or 'unknown'}"
    )

    parts.append(
        "## Headline metrics (1.0 = this video's own average)\n"
        f"  opening (first 2s)   {features.opening_2s:.3f}\n"
        f"  closing (last 2s)    {features.closing_2s:.3f}\n"
        f"  peak                 {features.peak_value:.3f} at {features.peak_time_s:.1f}s\n"
        f"  trough               {features.trough_value:.3f} at {features.trough_time_s:.1f}s\n"
        f"  swing                {features.swing:.3f}\n"
        f"  variability          {features.variability:.3f}\n"
        f"  decay (open-close)   {features.decay:+.3f}\n"
        f"  time above average   {features.sustained_above:.1%}"
    )
    if features.tail_spike:
        parts.append(
            "NOTE: the final timestep spikes. This is almost certainly a hard cut "
            "(to black or a card), not the ending landing. Do not treat it as a finding."
        )

    if features.segments:
        seg_lines = "\n".join(
            f"  {s.label:<18} {s.start_s:5.1f}-{s.end_s:5.1f}s  {s.mean:.3f}  {s.verdict}"
            for s in features.segments
        )
        parts.append(f"## Sections the user defined\n{seg_lines}")
    if segments_note:
        parts.append(segments_note)

    parts.append(f"## Response curve over time\n{_trace_table(features)}")

    if comparison:
        sec = "\n".join(
            f"  {s.label:<16} yours {s.subject:.3f}  ref {s.reference:.3f}  "
            f"{s.delta:+.3f}  {s.verdict}"
            for s in comparison.sections
        )
        parts.append(
            f"## Comparison against reference: {comparison.reference_label}\n"
            f"Shape correlation: {comparison.correlation:+.3f} "
            f"({'same arc' if comparison.same_shape else 'different arc'})\n"
            f"{sec}"
        )
        if comparison.notes:
            parts.append("Comparison caveats:\n" + "\n".join(f"  - {n}" for n in comparison.notes))

    if history and history.n_previous:
        pct = "\n".join(f"  {k:<18} {v:5.1f}th percentile" for k, v in history.percentiles.items())
        parts.append(
            f"## Position among the user's own {history.n_previous} previous evaluations\n{pct}"
        )
        if history.trend:
            parts.append("Trend across their history:\n" + "\n".join(
                f"  {k}: {v}" for k, v in history.trend.items()))

    if learned and learned.n_labelled:
        parts.append(f"## What their real published performance suggests\n{learned.summary_text()}")
        parts.append(f"IMPORTANT: {learned.caveat}")

    return "\n\n".join(parts)


def generate(
    cfg: Config,
    features: Features,
    video_label: str,
    comparison: Comparison | None = None,
    history: HistoryPosition | None = None,
    learned: learn.LearnedModel | None = None,
    extra_instruction: str = "",
) -> tuple[str, dict[str, Any]]:
    """Ask the LLM for recommendations. Returns (markdown, evidence)."""
    provider = get_provider(cfg)
    context = build_context(features, video_label, comparison, history, learned)

    user = (
        f"{context}\n\n"
        "---\n\n"
        "Here is what you must know about the measurement instrument, which you "
        "should reflect in how confidently you speak:\n\n"
        f"{explain.llm_context()}\n\n"
        "---\n\n"
        "Write the analysis now."
    )
    if extra_instruction:
        user += f"\n\nAdditional instruction from the user: {extra_instruction}"

    body = provider.complete(
        [Message("system", SYSTEM_PROMPT), Message("user", user)]
    )
    evidence = {
        "model": provider.model,
        "had_comparison": comparison is not None,
        "had_history": bool(history and history.n_previous),
        "labelled_videos": learned.n_labelled if learned else 0,
        "learning_usable": bool(learned and learned.usable),
        "modalities": features.modalities,
    }
    return body, evidence


RETRO_PROMPT = """\
You previously analysed this video. Since then, real published performance data
has been recorded for this and other videos. Revisit your advice.

Your job is NOT to repeat the original analysis. It is to say what has CHANGED
now that outcomes are known: which of the original recommendations look
supported, which look wrong, and what new advice the outcome data enables.

If the outcome data is too thin to change anything, say exactly that in two
sentences and stop. Do not manufacture revisions.\
"""


def generate_retroactive(
    cfg: Config,
    features: Features,
    video_label: str,
    previous: str,
    learned: learn.LearnedModel,
    own_performance: list[dict] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Revisit an old evaluation in the light of newly recorded outcomes."""
    provider = get_provider(cfg)
    context = build_context(features, video_label, learned=learned)

    perf_txt = ""
    if own_performance:
        perf_txt = "\n## This video's own published performance\n" + "\n".join(
            f"  {p}" for p in own_performance
        )

    user = (
        f"{RETRO_PROMPT}\n\n{context}{perf_txt}\n\n"
        f"## Your previous recommendations\n\n{previous}\n\n"
        "---\n\nWrite the revision now."
    )
    body = provider.complete([Message("system", SYSTEM_PROMPT), Message("user", user)])
    evidence = {
        "model": provider.model,
        "kind": "retroactive",
        "labelled_videos": learned.n_labelled,
        "learning_usable": learned.usable,
    }
    return body, evidence


def answer_question(
    cfg: Config,
    question: str,
    features: Features | None,
    video_label: str,
    comparison: Comparison | None,
    learned: learn.LearnedModel | None,
    chat_rows: list[Any] | None = None,
    stream: bool = True,
):
    """Interactive Q&A grounded in one evaluation (or the whole corpus)."""
    provider = get_provider(cfg)
    messages = [Message("system", SYSTEM_PROMPT + "\n\nYou are now answering follow-up "
                                                  "questions conversationally. Stay concise.")]
    if features:
        messages.append(
            Message("user", "Context for all following questions:\n\n"
                    + build_context(features, video_label, comparison, learned=learned)
                    + "\n\nMeasurement caveats:\n" + explain.llm_context())
        )
        messages.append(Message("assistant", "Understood. Ask away."))
    for row in chat_rows or []:
        messages.append(Message(row["role"], row["content"]))
    messages.append(Message("user", question))

    if stream:
        return provider.stream(messages)
    return provider.complete(messages)
