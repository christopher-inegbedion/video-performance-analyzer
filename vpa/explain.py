"""Plain-English explanations of what TRIBE v2 is and what its numbers mean.

A core design rule of this tool: never show a user a number without being able
to tell them what it is, what it is not, and how much to trust it. These strings
are surfaced in reports, in `vpa explain`, and injected into the LLM's context so
its recommendations inherit the same caveats.
"""

from __future__ import annotations

WHAT_IS_TRIBE = """\
TRIBE v2 is a brain-encoding model published by Meta AI. It was trained on fMRI
recordings from over 700 people watching and listening to naturalistic media, and
it learned to predict the brain response a stimulus would produce.

You give it a video. It gives back a predicted fMRI signal: one value for each of
20,484 points on the cortical surface, for every half-second of your video.

It is a neuroscience research instrument, not an advertising tool. It was built so
researchers could test hypotheses without putting a person in a scanner.\
"""

WHAT_IT_MEASURES = """\
This tool reduces that huge array to one number per timestep: the total predicted
cortical response, normalised so 1.0 is average *for that video*.

So the curve answers: "relative to itself, where does this video produce more
predicted neural response, and where less?"

That is a question about SHAPE. It is reliable for comparing moments within one
video, and for comparing the shape of two videos. It is much weaker as an
absolute score — a video's mean is 1.0 by construction, always.\
"""

WHAT_IT_IS_NOT = """\
TRIBE predicts fMRI response. It does NOT predict:

  * attention, watch time, or whether someone keeps scrolling
  * clicks, conversions, sales, or brand recall
  * whether people like your video

Treating "more predicted activation" as "better advert" is an inference the model
does not make and its authors do not claim. High response can mean engagement —
it can equally mean confusion, surprise, or a jarring edit. The model cannot tell
those apart, and neither can this tool.

Use it to rank variants of the same idea against each other. Do not use it as a
verdict on a single video in isolation.\
"""

KNOWN_ARTEFACTS = """\
Things that reliably spike the curve for uninteresting reasons:

  * A hard cut — to black, to a title card, to a wildly different image. Visual
    discontinuity produces a large response regardless of content. This tool
    flags a final-step spike as `tail_spike` and excludes the tail when hunting
    for peaks, because a cut to black is not your ending landing.
  * A change of mode — photographs to typography, silence to speech. The novelty
    is doing the work, not the material.
  * Position, not content. In testing, the SAME images scored highest in one edit
    and lowest in another purely because of where they sat. If you move a shot
    and its score changes, that is expected — it does not mean the shot got
    better or worse.\
"""

HAEMODYNAMIC_NOTE = """\
fMRI measures blood flow, which lags neural activity by several seconds. TRIBE's
predictions are offset by 5 seconds to compensate.

For a 10-30 second video this matters: the model is describing a response that is
still unfolding after your video has ended. Fine detail near the very end of a
short film should be read with that in mind.\
"""

MODALITY_NOTE = """\
TRIBE is tri-modal — vision, audition and language.

  * Vision always runs.
  * Audition runs when your video has a soundtrack.
  * Language runs only when speech is transcribed into word events, which needs
    the gated meta-llama/Llama-3.2-1B model and a HuggingFace token.

By default this tool runs vision + audition, which needs no licence acceptance.
That means the model hears the SOUND of speech — its rhythm, pacing and where it
sits against silence — but does not process what the words MEAN.

This is not a small difference. In testing, the same video scored silent versus
with-sound put its weakest section up by more than 0.1 normalised units. If you
compare two evaluations, make sure they used the same modalities.\
"""

METRIC_GLOSSARY = {
    "opening_2s": "Mean response over the first two seconds. On a feed this is the "
                  "only part of the curve that is definitely doing work, because the "
                  "scroll decision happens before anything else matters.",
    "closing_2s": "Mean response over the last two seconds. Watch for tail_spike — a "
                  "cut to black inflates this.",
    "peak_value": "Highest normalised response, excluding the tail. Where the video "
                  "produces most predicted response.",
    "peak_time_s": "When that peak happens, in seconds.",
    "trough_value": "Lowest normalised response. Often the most actionable number in "
                    "the report — it is where the edit sags.",
    "trough_time_s": "When the sag happens.",
    "swing": "Peak minus trough. High swing means a dynamic film; low swing means an "
             "even one. Neither is automatically better.",
    "variability": "Standard deviation of the curve. A flat film has low variability.",
    "decay": "Opening minus closing. Positive means the video sheds response as it "
             "runs. Most videos decay; ending higher than you started is unusual.",
    "sustained_above": "Fraction of the video spent above its own average.",
    "mean_magnitude": "Raw un-normalised magnitude. Only comparable between videos "
                      "scored with identical modalities and frame rate.",
    "correlation": "Shape similarity between your video and the reference, -1 to 1. "
                   "Above ~0.85 means the two follow essentially the same arc.",
}

LICENCE_WARNING = """\
TRIBE v2's weights are released under CC BY-NC 4.0 — NON-COMMERCIAL.

Using it to optimise commercial advertising is arguably outside that licence.
This tool will not stop you, and it is not legal advice; it is a decision worth
making deliberately rather than drifting into.\
"""

SECTIONS = {
    "what": ("What TRIBE v2 is", WHAT_IS_TRIBE),
    "measures": ("What the numbers mean", WHAT_IT_MEASURES),
    "limits": ("What it is NOT", WHAT_IT_IS_NOT),
    "artefacts": ("Known artefacts", KNOWN_ARTEFACTS),
    "lag": ("Haemodynamic lag", HAEMODYNAMIC_NOTE),
    "modalities": ("Modalities", MODALITY_NOTE),
    "licence": ("Licence", LICENCE_WARNING),
}


def metric_help(name: str) -> str:
    return METRIC_GLOSSARY.get(name, "No explanation recorded for this metric.")


def llm_context() -> str:
    """The caveats the model must inherit when writing recommendations."""
    return "\n\n".join(
        [WHAT_IS_TRIBE, WHAT_IT_MEASURES, WHAT_IT_IS_NOT, KNOWN_ARTEFACTS, MODALITY_NOTE]
    )


def full_text() -> str:
    parts = []
    for _, (title, body) in SECTIONS.items():
        parts.append(f"## {title}\n\n{body}")
    glossary = "\n".join(f"  {k:<18} {v}" for k, v in METRIC_GLOSSARY.items())
    parts.append(f"## Metric glossary\n\n{glossary}")
    return "\n\n".join(parts)
