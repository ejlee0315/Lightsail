"""Logging configuration for experiment runs.

Installs two handlers on the root logger:

- a console ``StreamHandler`` at the requested level (INFO by default),
- a ``FileHandler`` writing everything at DEBUG level to
  ``<output_dir>/run.log``.

Call :func:`setup_logging` exactly once per run. Subsequent calls on the
same run clear the previous handlers so re-running inside a REPL or test
does not spam duplicate log lines.
"""

from __future__ import annotations

import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    output_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """Initialize console + file logging for one experiment run.

    Returns the path to the log file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    root = logging.getLogger()

    # Remove any previously-attached handlers (e.g. from a prior run).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover
            pass

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.setLevel(min(console_level, file_level))

    # Tame noisy third-party loggers.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("botorch").setLevel(logging.WARNING)
    logging.getLogger("gpytorch").setLevel(logging.WARNING)
    logging.getLogger("linear_operator").setLevel(logging.WARNING)

    return log_file
