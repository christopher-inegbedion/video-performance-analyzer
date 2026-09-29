"""Keep third-party output from corrupting the progress display.

Two separate problems, both of which made a long run hard to follow:

  1. NOISE. Importing tribev2 pulls in neuralset, which emits warnings about
     label encoders on every import. huggingface_hub prints its own download
     bars and an unauthenticated-request warning. None of it is actionable and
     all of it prints straight over the top of the live progress display.

  2. SILENCE. The encoder is the slow part — tens of minutes — and it reports
     progress through its own tqdm bar. If that bar is suppressed, the stage
     sits frozen at one percentage for the whole run and looks like a hang. So
     rather than hide it, we intercept it and feed the real numbers into our
     own display.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterator

# stage label, completed units, total units
TqdmFn = Callable[[str, int, int], None]

_NOISY_LOGGERS = (
    "neuralset",
    "neuralset.extractors",
    "neuralset.extractors.base",
    "neuraltrain",
    "exca",
    "huggingface_hub",
    "transformers",
)


@contextlib.contextmanager
def quiet_imports() -> Iterator[None]:
    """Silence import-time chatter. Real errors still propagate."""
    previous = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Set before huggingface_hub is imported so its bars never start.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neuralset.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            yield
        finally:
            for name, level in previous.items():
                logging.getLogger(name).setLevel(level)


@contextlib.contextmanager
def relay_tqdm(on_update: TqdmFn | None) -> Iterator[None]:
    """Route third-party tqdm bars into our own progress display.

    Without this the encoder's bar either fights the live display for the
    terminal, or is hidden and the user watches a frozen percentage for half an
    hour. Patched at the module level because TRIBE resolves tqdm at call time.
    """
    if on_update is None:
        yield
        return

    try:
        import tqdm as tqdm_mod
        import tqdm.auto as tqdm_auto
    except ModuleNotFoundError:  # pragma: no cover - tqdm ships with TRIBE
        yield
        return

    original = tqdm_mod.tqdm
    original_auto = tqdm_auto.tqdm

    class _Relaying(original):  # type: ignore[misc,valid-type]
        """Forwards progress to our display instead of drawing its own bar.

        Note: we deliberately do NOT pass disable=True. tqdm short-circuits
        update() when disabled and never advances its counter, which would make
        every relayed value 0 — a frozen bar, which is the bug this whole class
        exists to fix. Its own drawing is harmless because stdout/stderr are
        already diverted by captured_output().
        """

        def __init__(self, *args, **kwargs):
            self._relay_n = 0
            super().__init__(*args, **kwargs)
            self._report()

        def _report(self) -> None:
            total = getattr(self, "total", None)
            if not total:
                return
            with contextlib.suppress(Exception):
                on_update(
                    str(getattr(self, "desc", "") or "working"),
                    int(self._relay_n),
                    int(total),
                )

        def update(self, n: int = 1):  # noqa: D102
            result = super().update(n)
            self._relay_n += n or 0
            self._report()
            return result

    # Patching tqdm.tqdm alone is NOT enough, and this is the trap that made an
    # earlier version of this relay silently do nothing.
    #
    # neuralset does `from tqdm import tqdm` at module import time, which binds
    # the original class into its own namespace. Rebinding the attribute on the
    # tqdm module never reaches that copy, so the encoder kept using the real
    # tqdm and our display sat frozen for the whole run.
    #
    # So: rebind the name in every module that already holds a reference to it.
    rebound: list[tuple[object, str]] = []
    for module in list(sys.modules.values()):
        if module is None or module is tqdm_mod or module is tqdm_auto:
            continue
        for attr in ("tqdm",):
            with contextlib.suppress(Exception):
                if getattr(module, attr, None) in (original, original_auto):
                    setattr(module, attr, _Relaying)
                    rebound.append((module, attr))

    tqdm_mod.tqdm = _Relaying
    tqdm_auto.tqdm = _Relaying
    try:
        yield
    finally:
        for module, attr in rebound:
            with contextlib.suppress(Exception):
                setattr(module, attr, original)
        tqdm_mod.tqdm = original
        tqdm_auto.tqdm = original_auto


@contextlib.contextmanager
def captured_output(sink: list[str] | None = None) -> Iterator[io.StringIO]:
    """Divert stdout/stderr into a buffer for the duration of a block.

    Suppressing individual loggers and warning filters is not enough: TRIBE and
    its dependencies write at several different points — snapshot_download, model
    loading, event extraction — and any one of them printing over a live progress
    display makes a long run unreadable.

    So the whole model section runs with output diverted. The text is kept, not
    thrown away: it is handed back for `--verbose` and printed in full if the run
    fails, because that is exactly when someone needs it.
    """
    buffer = io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    root = logging.getLogger()
    previous_level = root.level
    sys.stdout = buffer
    sys.stderr = buffer
    root.setLevel(logging.ERROR)

    # Handlers created at import time hold their own reference to the real
    # stderr, so swapping sys.stderr never reaches them. Repoint each one.
    # Walk every registered logger, not a hand-written list: third-party code
    # attaches handlers under names we cannot predict, and any one of them
    # printing mid-run corrupts the display.
    rerouted: list[tuple[logging.StreamHandler, object]] = []
    every = [root, *logging.root.manager.loggerDict.values()]
    for logger in every:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                rerouted.append((handler, handler.stream))
                with contextlib.suppress(Exception):
                    handler.setStream(buffer)
    try:
        yield buffer
    finally:
        for handler, original_stream in rerouted:
            with contextlib.suppress(Exception):
                handler.setStream(original_stream)
        sys.stdout = real_out
        sys.stderr = real_err
        root.setLevel(previous_level)
        if sink is not None:
            text = buffer.getvalue().strip()
            if text:
                sink.append(text)
