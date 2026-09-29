# video-performance-analyzer

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
pip install 'video-performance-analyzer[tribe]' # + scoring dependencies
pip install git+https://github.com/facebookresearch/tribev2.git
```

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
vpa show eval_a1b2c3                      # revisit a past evaluation
vpa list evals                            # everything you have run
vpa ask eval_a1b2c3                       # ask questions about it
vpa explain artefacts                     # what the numbers can't tell you
vpa tui                                   # interactive session
```

### Describing your structure

Even segments are a poor guide. Tell it where your real beats are and the report
speaks your language:

```bash
vpa analyse cut.mp4 --segments "hook:0-3,montage:3-16,card:16-19,logo:19-24"
```

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

Expect a few minutes per 30-second video on CPU.

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

Contributions welcome — particularly platform connectors for metrics ingestion
(there is a documented seam in `vpa/metrics.py`) and additional LLM providers
(`vpa/providers/`).
