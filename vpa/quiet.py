"""Silence third-party import noise the user cannot act on.

Importing tribev2 pulls in neuralset, which emits a UserWarning and a logging
WARNING about label encoders on every import. They are not actionable and not
about the user's video, so they are suppressed at the boundary rather than left
to corrupt otherwise-clean CLI output. Real errors still propagate.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import Iterator


@contextlib.contextmanager
def quiet_imports() -> Iterator[None]:
    noisy = ("neuralset", "neuralset.extractors", "neuralset.extractors.base",
             "neuraltrain", "exca")
    previous = {name: logging.getLogger(name).level for name in noisy}
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neuralset.*")
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            yield
        finally:
            for name, level in previous.items():
                logging.getLogger(name).setLevel(level)
