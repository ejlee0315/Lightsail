"""Initial-phase samplers for the BO runner.

Two methods are supported:

- ``sobol``: quasi-random low-discrepancy sequence via ``torch.quasirandom.SobolEngine``.
  Requires torch (which is already a hard dep of the ``mobo`` extra).
- ``lhs``: classical Latin Hypercube with one uniform jitter per cell.
  Uses only numpy, so it works without torch.

Both return a ``(n, d)`` numpy array with values in ``[0, 1]``.
A fixed integer ``seed`` makes them reproducible.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sobol
# ---------------------------------------------------------------------------


def sobol_samples(
    n: int,
    d: int,
    seed: int = 42,
    scramble: bool = True,
) -> np.ndarray:
    """Draw ``n`` Sobol samples in ``[0, 1]^d``.

    Requires torch. Raises ``ImportError`` if torch is not available.
    """
    try:
        from torch.quasirandom import SobolEngine
    except ImportError as err:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "sobol_samples requires torch; install with `pip install 'lightsail[mobo]'`"
        ) from err

    engine = SobolEngine(dimension=d, scramble=scramble, seed=seed)
    samples = engine.draw(n).numpy().astype(np.float64)
    return samples


# ---------------------------------------------------------------------------
# Latin Hypercube
# ---------------------------------------------------------------------------


def latin_hypercube(
    n: int,
    d: int,
    seed: int = 42,
) -> np.ndarray:
    """Classical Latin Hypercube sampling in ``[0, 1]^d``.

    For each dimension independently, the unit interval is split into ``n``
    sub-intervals, each receives exactly one jittered sample, and the
    column is then randomly permuted. No torch required.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        jitter = rng.uniform(0.0, 1.0, size=n)
        result[:, j] = (perm + jitter) / n
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def initial_samples(
    n: int,
    d: int,
    method: str = "sobol",
    seed: int = 42,
) -> np.ndarray:
    """Dispatch to the requested initial sampler.

    Valid ``method`` strings: ``"sobol"``, ``"lhs"``, ``"latin_hypercube"``.
    """
    m = method.lower()
    if m == "sobol":
        return sobol_samples(n, d, seed=seed)
    if m in ("lhs", "latin_hypercube"):
        return latin_hypercube(n, d, seed=seed)
    raise ValueError(f"Unknown sampling method: {method!r}")
