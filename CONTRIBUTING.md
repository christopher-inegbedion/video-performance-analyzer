# Contributing

Thanks for looking. This is a small project with a specific point of view, so
this file covers what that is — it should save you guessing.

## Setup

```bash
git clone https://github.com/christopher-inegbedion/video-performance-analyzer.git
cd video-performance-analyzer
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest -q
ruff check vpa tests
```

That is enough for everything except actually scoring a video. For that you also
need ffmpeg, roughly 3GB of model weights and a lot of patience:

```bash
pip install -e '.[tribe]'
pip install git+https://github.com/facebookresearch/tribev2.git
vpa doctor          # says exactly what is missing
```

**CI deliberately does not install the scoring extras.** A clean core install is
what catches undeclared dependencies — that is how an accidental reliance on
`tqdm` was found, after it passed locally for days. If you add something to
`vpa/` that imports a new package, declare it in `pyproject.toml`.

## The one rule that matters

**Never show a number without being able to say what it means and what it does
not.**

TRIBE v2 predicts fMRI response. It does not predict attention, watch time,
clicks or sales. The whole tool is built around not letting that slide:

- every metric has an entry in `vpa/explain.py`, and a test asserts it
- the same caveats are injected into the LLM's context, so recommendations
  inherit them rather than overclaiming
- known artefacts are handled, not reported as insight — a cut to black spikes
  the curve every time, so the tail is excluded from peak detection

If you add a metric, add its explanation in the same change. `tests/test_explain.py`
will fail if you do not.

## Where things live

| Module | Responsibility |
|---|---|
| `vpa/tribe.py` | Runs TRIBE and patches what stops it working off a GPU cluster |
| `vpa/analysis/` | Predictions → response curve → comparisons |
| `vpa/learn.py` | Correlating predicted features with real published performance |
| `vpa/recommend.py` | Prompt construction; the rules the model must obey |
| `vpa/providers/` | LLM backends |
| `vpa/explain.py` | What everything means, and its limits |
| `vpa/quiet.py` | Keeping third-party output from wrecking the progress display |

## Good first contributions

**Platform connectors for metrics.** Performance is entered by hand or via CSV
today. `vpa/metrics.py` has a documented seam: anything that can produce
`{video, views, likes, ...}` rows can feed `record()`. Instagram Graph API,
TikTok, YouTube Analytics are all wanted. Keep manual entry working — it is the
only thing that covers every platform.

**LLM backends.** `vpa/providers/` has a `Provider` protocol and one
OpenAI-compatible implementation covering most services. Anything with a
different shape (Anthropic's native API, Bedrock, Vertex) needs its own class
registered in `get_provider`.

**Speed.** Scoring runs at roughly 96x realtime on CPU. Lowering `target_fps`
helps proportionally, but nobody has measured what it costs in accuracy. A
careful comparison of the same video at 24 and 12 fps would be genuinely
valuable and needs no new code.

**Honesty about small samples.** `vpa/learn.py` refuses to call correlations
findings below five labelled videos. That threshold is a guess. If you know the
statistics better than I do, improve it.

## Style

- Comments explain *why*, especially where the code looks odd. Most of
  `vpa/tribe.py` is strange for a reason and each reason is written down.
- Errors should say what to do next. `vpa doctor` is the model: state what is
  missing and give the command that fixes it.
- Prefer failing loudly over guessing quietly. A silent wrong number is worse
  than an error.
- `ruff check vpa tests` must pass. Line length 100.

## Testing

`pytest -q` needs no models, no API key and no network. Anything requiring those
belongs behind a skip.

Test behaviour rather than implementation. The most valuable test in the suite
checks that the progress relay reports *climbing* values — it caught a bug where
every relayed value was zero, which would have shipped a frozen progress bar as
a fix for a frozen progress bar.

## Licences

This tool is MIT. The models are not. TRIBE v2 is CC BY-NC 4.0 — non-commercial
— and the language pathway needs a gated Llama repo. Please do not add anything
that obscures those terms from users; the current behaviour of stating them
plainly is deliberate.
