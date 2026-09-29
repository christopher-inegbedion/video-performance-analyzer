"""Configuration: TOML file + environment overrides.

Everything the tool does is meant to be customisable without touching code.
Config lives at ~/.config/vpa/config.toml (or $VPA_CONFIG).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import platformdirs
import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib  # type: ignore


def config_dir() -> Path:
    override = os.environ.get("VPA_CONFIG_DIR")
    return Path(override) if override else Path(platformdirs.user_config_dir("vpa"))


def data_dir() -> Path:
    override = os.environ.get("VPA_DATA_DIR")
    return Path(override) if override else Path(platformdirs.user_data_dir("vpa"))


def config_path() -> Path:
    override = os.environ.get("VPA_CONFIG")
    return Path(override) if override else config_dir() / "config.toml"


@dataclass
class LLMConfig:
    """Provider-agnostic LLM settings. Defaults to OpenRouter."""

    provider: str = "openrouter"
    # A capable, inexpensive default. Any OpenRouter model id works here, e.g.
    #   google/gemini-2.5-flash · openai/gpt-4o-mini
    #   deepseek/deepseek-chat · meta-llama/llama-3.3-70b-instruct
    model: str = "google/gemini-2.5-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 2000
    temperature: float = 0.3
    timeout_s: int = 120
    # Optional OpenRouter attribution headers
    referer: str = "https://github.com/yourname/video-performance-analyzer"
    title: str = "video-performance-analyzer"


@dataclass
class TribeConfig:
    """How to run TRIBE v2 locally."""

    checkpoint: str = "facebook/tribev2-mini"  # or facebook/tribev2 (larger, slower)
    device: str = "auto"  # auto | cpu | cuda
    # The encoder normalises to 256px anyway; downscaling first is a large speedup
    # with no meaningful change to what the model sees.
    scale_width: int = 360
    scale_height: int = 640
    # Frame rate matters a lot: the encoder works in fixed-length frame clips, so a
    # 60fps source costs ~2.5x a 24fps one for the same seconds of footage.
    target_fps: int = 24
    # Language pathway needs a gated Llama repo + HF token. Off by default so the
    # tool works out of the box with vision + audition only.
    enable_language: bool = False
    whisper_model: str = "base.en"  # local word timings, avoids whisperx dependency hell
    cache_dir: str = ""  # empty -> <data_dir>/tribe-cache


@dataclass
class AnalysisConfig:
    """Thresholds used when turning a response curve into findings."""

    # A section is called 'strong'/'weak' when it deviates this far from the
    # video's own mean (1.0 == exactly average).
    strong_threshold: float = 1.10
    weak_threshold: float = 0.90
    # Ignore the final N seconds when hunting for peaks: a cut to black is a hard
    # discontinuity and always spikes, which is an artefact, not an insight.
    ignore_tail_s: float = 1.5
    # Minimum correlation between two curves before we call them "the same shape".
    same_shape_r: float = 0.85


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tribe: TribeConfig = field(default_factory=TribeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def tribe_cache(self) -> Path:
        if self.tribe.cache_dir:
            return Path(self.tribe.cache_dir)
        return data_dir() / "tribe-cache"

    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> Config:
    """Load config, falling back to defaults for anything unset."""
    defaults = asdict(Config())
    path = config_path()
    if path.exists():
        with path.open("rb") as fh:
            defaults = _merge(defaults, tomllib.load(fh))
    cfg = Config(
        llm=LLMConfig(**defaults.get("llm", {})),
        tribe=TribeConfig(**defaults.get("tribe", {})),
        analysis=AnalysisConfig(**defaults.get("analysis", {})),
    )
    # Environment always wins, so CI and one-off runs need no file edits.
    if m := os.environ.get("VPA_MODEL"):
        cfg.llm.model = m
    if b := os.environ.get("VPA_BASE_URL"):
        cfg.llm.base_url = b
    if d := os.environ.get("VPA_DEVICE"):
        cfg.tribe.device = d
    return cfg


def write_default(path: Path | None = None, force: bool = False) -> Path:
    """Write a commented config file the user can edit."""
    target = path or config_path()
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    body = tomli_w.dumps(asdict(Config()))
    header = (
        "# video-performance-analyzer configuration\n"
        "# Every value here is optional; delete a line to fall back to the default.\n"
        "#\n"
        "# llm.model    any OpenRouter model id (or any OpenAI-compatible endpoint\n"
        "#              if you change llm.base_url — Ollama, vLLM, LiteLLM all work)\n"
        "# tribe.*      how the brain-encoding model runs locally\n"
        "# analysis.*   thresholds for turning curves into findings\n\n"
    )
    target.write_text(header + body)
    return target
