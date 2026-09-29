"""The output-handling layer is what makes a 40-minute run followable."""

from __future__ import annotations

import logging
import sys

from vpa.quiet import captured_output, quiet_imports, relay_tqdm


def test_captured_output_diverts_prints():
    with captured_output() as buf:
        print("noisy library message")
        print("to stderr", file=sys.stderr)
    assert "noisy library message" in buf.getvalue()
    assert "to stderr" in buf.getvalue()


def test_captured_output_restores_streams():
    before_out, before_err = sys.stdout, sys.stderr
    with captured_output():
        pass
    assert sys.stdout is before_out
    assert sys.stderr is before_err


def test_captured_output_catches_preexisting_log_handlers():
    """Handlers created before the capture hold their own stream reference.

    This is the case that leaked in practice: a library configures its logger at
    import time, so swapping sys.stderr never reaches it.
    """
    logger = logging.getLogger("vpa_test_preexisting")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)  # bound to the REAL stderr now
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    with captured_output() as buf:
        logger.warning("this must not reach the terminal")

    assert "this must not reach the terminal" in buf.getvalue()
    # and the handler must be given its original stream back
    assert handler.stream is sys.stderr
    logger.handlers.clear()


def test_captured_output_appends_to_sink():
    sink: list[str] = []
    with captured_output(sink):
        print("kept for --verbose")
    assert sink and "kept for --verbose" in sink[0]


def test_relay_tqdm_reports_real_progress():
    """The encode is the slow part; its progress must reach our display."""
    import tqdm

    seen: list[tuple[str, int, int]] = []
    with relay_tqdm(lambda desc, done, total: seen.append((desc, done, total))):
        bar = tqdm.tqdm(total=10, desc="Encoding video")
        for _ in range(10):
            bar.update(1)
        bar.close()

    assert seen, "no progress was relayed"
    assert seen[-1][1] == 10 and seen[-1][2] == 10
    assert "Encoding" in seen[-1][0]
    # It must climb, not just report the endpoints.
    assert [s[1] for s in seen] == sorted(s[1] for s in seen)


def test_relay_tqdm_restores_tqdm():
    import tqdm

    original = tqdm.tqdm
    with relay_tqdm(lambda *a: None):
        assert tqdm.tqdm is not original
    assert tqdm.tqdm is original


def test_relay_tqdm_with_no_callback_is_a_noop():
    import tqdm

    original = tqdm.tqdm
    with relay_tqdm(None):
        assert tqdm.tqdm is original


def test_quiet_imports_restores_logger_levels():
    logger = logging.getLogger("neuralset")
    logger.setLevel(logging.DEBUG)
    with quiet_imports():
        assert logger.level == logging.ERROR
    assert logger.level == logging.DEBUG


def test_relay_reaches_modules_that_imported_tqdm_early(monkeypatch):
    """The failure this relay actually shipped with.

    Libraries do `from tqdm import tqdm` at import time, binding the original
    class into their own namespace. Patching `tqdm.tqdm` never reaches that
    copy, so the encoder kept drawing its own bar and our display stayed frozen
    for the entire run — while a test that called `tqdm.tqdm()` directly passed.

    This test imitates the real import pattern instead.
    """
    import sys
    import types

    import tqdm as tqdm_mod

    # A stand-in for neuralset.extractors.video: grabs tqdm at import time.
    fake = types.ModuleType("vpa_fake_library")
    fake.tqdm = tqdm_mod.tqdm  # the binding that defeated the old patch
    monkeypatch.setitem(sys.modules, "vpa_fake_library", fake)

    seen: list[int] = []
    with relay_tqdm(lambda desc, done, total: seen.append(done)):
        # The library calls ITS bound reference, not tqdm.tqdm
        bar = fake.tqdm(total=4, desc="Encoding video")
        for _ in range(4):
            bar.update(1)
        bar.close()

    assert seen, "progress from an early-bound tqdm never reached the display"
    assert seen[-1] == 4

    # and the library's own reference must be handed back
    assert fake.tqdm is tqdm_mod.tqdm
