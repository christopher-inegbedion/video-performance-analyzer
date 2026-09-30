# video-performance-analyzer

[![PyPI](https://img.shields.io/pypi/v/video-performance-analyzer)](https://pypi.org/project/video-performance-analyzer/)
[![Python](https://img.shields.io/pypi/pyversions/video-performance-analyzer)](https://pypi.org/project/video-performance-analyzer/)
[![test](https://github.com/christopher-inegbedion/video-performance-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/christopher-inegbedion/video-performance-analyzer/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predict how a video lands before you publish it — then find out whether the
prediction was right, and get better advice because of it.

`vpa` runs Meta's **TRIBE v2** brain-encoding model over your video, reduces the
output to a response curve you can read, compares it against a reference video
and against your own back catalogue, and asks an LLM for concrete editing
recommendations. When you later record what the video actually did — views,
likes, watch-through — it learns, and goes back to revise advice it has already
given you.

```
vpa analyse my-cut-v3.mp4 --reference competitor-ad.mp4 \
    --segments "hook:0-3,montage:3-16,card:16-19,end:19-24"
```

```
╭─ my-cut-v3 ──────────────────────────────────────────────────╮
│ 24.1s · 25 timesteps · Audio, Video                          │
╰──────────────────────────────────────────────────────────────╯
╭─ predicted response over time ───────────────────────────────╮
│ ▃▄▄▅▅▆▇▇▆▇▇▇▆▅▄▃▃▃▃▂▂▂▂█                                     │
│ 0s                  24s                                      │
╰──────────────────────────────────────────────────────────────╯
  opening_2s           0.882   Mean response over the first two seconds.
  trough_value         0.777   Lowest normalised response.
  decay               +0.055   Opening minus closing.
```

---

## What this actually measures — read this first

TRIBE v2 predicts **fMRI brain response**. It does not predict attention, watch
time, clicks, or sales. Treating "more predicted response" as "better advert" is
an inference the model does not make and its authors do not claim.

This matters enough that the tool is built around it:

- Every metric shown has a plain-English explanation attached (`vpa explain`).
- The same caveats are injected into the LLM's context, so recommendations
  inherit them rather than overclaiming.
- Known artefacts are flagged automatically. A cut to black spikes the curve
  every time; `vpa` detects that and excludes it from peak-finding rather than
  reporting it as your ending landing.
- Position matters more than content. In testing, the *same images* scored
  highest in one edit and lowest in another purely because of where they sat.

Use it to rank variants of the same idea. Don't use it as a verdict on one video.

## Install

```bash
pip install video-performance-analyzer          # core
pip install 'video-performance-analyzer[tribe]' # + scoring dependencies (~3GB of models)
pip install git+https://github.com/facebookresearch/tribev2.git
```

TRIBE itself is not on PyPI, so that last line is always needed for scoring.

<details>
<summary>From source instead</summary>

```bash
git clone https://github.com/christopher-inegbedion/video-performance-analyzer.git
cd video-performance-analyzer
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```
</details>

You also need **ffmpeg** (`brew install ffmpeg` / `apt install ffmpeg`) and an
API key for whichever LLM you point it at:

```bash
export OPENROUTER_API_KEY=sk-or-...
vpa doctor      # tells you exactly what is missing
```

## Use

```bash
vpa analyse cut.mp4                       # score one video
vpa analyse cut.mp4 -r reference.mp4      # compare against something
vpa analyse https://tiktok.com/@you/...   # fetch a published post and score it
vpa show eval_a1b2c3                      # revisit a past evaluation
vpa list evals                            # everything you have run
vpa ask eval_a1b2c3                       # ask questions about it
vpa explain artefacts                     # what the numbers can't tell you
vpa tui                                   # interactive session
```

### Seeing the moments, not just the numbers

Every run extracts the actual frames at the moments the curve flags — the
opening, the peak, the trough, the middle of each section you defined — and
tiles them into one contact sheet.

This is the difference between a report and a decision. "Your trough is at
14.4s" makes you go and scrub through your own edit; the frame at 14.4s tells
you what to change.

```
key moments — the actual frames
  moment                 at   value  frame
  opening              0.5s   1.042  opening_000.50s.jpg
  peak                 2.4s   1.157  peak_002.40s.jpg
  trough               4.6s   0.696  trough_004.60s.jpg
```

Markdown exports embed the images inline. Use `--no-frames` to skip extraction.

### Describing your structure

Even segments are a poor guide. Tell it where your real beats are and the report
speaks your language:

```bash
vpa analyse cut.mp4 --segments "hook:0-3,montage:3-16,card:16-19,logo:19-24"
```

### Analysing a published post

Anywhere a file path is accepted, a link works too — TikTok, YouTube, Instagram,
anything `yt-dlp` can read:

```bash
vpa analyse "https://www.tiktok.com/@you/video/7412345"
vpa analyse new-cut.mp4 -r "https://www.tiktok.com/@rival/video/7409999"
```

The video is downloaded and scored exactly as a local file would be. The useful
part is what comes with it: a published post carries its own view, like, comment
and share counts, and those are **recorded automatically** as that video's real
performance. One command both scores a post and files the outcome the tool
learns from.

Stats keep accruing after a post goes up, so a number read an hour after
publishing is not comparable with one read a week later. Re-read them whenever
you like:

```bash
vpa metrics sync                  # refresh every video that came from a link
vpa metrics sync my-cut-v3        # or just one
```

Picking a fixed horizon — say 72 hours — and syncing then is what turns a pile
of snapshots into a series you can actually fit a model against.

Fetching needs `yt-dlp`, which is an optional extra:

```bash
pip install 'video-performance-analyzer[fetch]'
```

Without it, links fail with an install hint and local files keep working. A post
that returns views but no like count is reported rather than recorded, because
views alone cannot be scored.

### The learning loop

This is what makes the tool improve. Record what a published video did:

```bash
vpa metrics add my-cut-v3 --views 12400 --likes 380 --platform instagram
vpa metrics template -o metrics.csv && vpa metrics import metrics.csv
vpa metrics show          # what it has learned so far
```

Once outcomes exist, two things change:

1. New recommendations weight **real performance above predicted response**.
2. Old evaluations become *stale* — their advice predates what you now know.
   `vpa retrofit` revisits them and writes a new generation of recommendations
   saying what changed. Nothing is overwritten; you can read every generation
   with `vpa show <id> -g 1`.

The tool is deliberately honest about sample size. Below five labelled videos it
refuses to call correlations findings and says so in plain terms.

## Configuration

```bash
vpa config init      # writes a commented config file
vpa config show      # effective settings and where they came from
```

Any OpenAI-compatible endpoint works — OpenRouter (default), OpenAI, Together,
Groq, or a local model:

```toml
[llm]
model    = "google/gemini-2.5-flash"
base_url = "https://openrouter.ai/api/v1"

[tribe]
target_fps = 24        # 60fps costs ~2.5x for identical footage
enable_language = false # true needs a gated Llama repo + HF token

[analysis]
ignore_tail_s = 1.5    # don't let the cut-to-black spike become a "finding"
```

For a fully local setup, point `base_url` at Ollama (`http://localhost:11434/v1`)
and no API key is needed.

## Running TRIBE on a laptop

TRIBE ships configured for Meta's GPU cluster. `vpa` patches the known problems
automatically and tells you what it changed:

| Problem | What happens without the fix |
|---|---|
| `device: cuda` baked into the checkpoint | `Torch not compiled with CUDA enabled` |
| `num_workers: 20` baked in | DataLoader workers die silently; the process sits at 3% CPU looking alive |
| `compute_type` hardcoded to `float16` | CPU speech extraction always fails |
| `uvx whisperx` resolves a broken torchaudio | `module 'torchaudio' has no attribute 'list_audio_backends'` |
| Language pathway needs gated `meta-llama/Llama-3.2-1B` | 401 on an otherwise working run |

The last two are why word timings are generated locally with faster-whisper and
written to the `.tsv` cache TRIBE reads — whisperx is never invoked. The language
pathway is **off by default**, so vision + audition work with no licence gate.

### How long it takes

Scoring is dominated entirely by the video encoder. Setup — importing the
package, loading the checkpoint, building events — is about 8 seconds. Everything
after that scales with how much footage you feed it.

Measured on an Apple Silicon laptop (CPU only), roughly **96x realtime**:

| video length | time to score |
|---:|---:|
| 10s | ~16 min |
| 15s | ~24 min |
| 24s | ~38 min |
| 60s | ~96 min |

**This is not an interactive tool.** Start a run and come back to it.

Three ways to make it tractable:

- **Halve the frame rate.** `target_fps = 12` roughly halves the encode, because
  cost is proportional to frames. V-JEPA samples frames rather than reading every
  one, so the quality cost is plausibly small — but that is untested, so measure
  it on your own material before trusting it.
- **Score an excerpt.** For a feed asset the first 6-10 seconds is where the
  scroll decision happens. Scoring only the opening is a legitimate strategy.
- **Use a GPU.** This is what the model was built for and it is a different order
  of magnitude. Set `device = "cuda"`.

There is no caching or warm-start trick that helps: the cost is the encoder, not
the setup.

## Licences

This tool is MIT. The models are not:

- **TRIBE v2** is **CC BY-NC 4.0 — non-commercial**. Using it to optimise
  commercial advertising is arguably outside that licence. That is your call to
  make deliberately, and the tool says so rather than hiding it.
- The language pathway uses **Llama-3.2-1B**, which is gated and carries Meta's
  own terms.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check vpa
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Platform
connectors for metrics ingestion (`vpa/metrics.py` has a documented seam) and
additional LLM providers (`vpa/providers/`) are the most useful places to start.
