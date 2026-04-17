"""Lattice family generators for the central PhC reflector.

Each lattice is a small class with three core methods:

- ``generate_sites(extent_nm)``: list of (x, y) positions inside a
  circular patch of the given diameter.
- ``nearest_neighbor_distance()``: smallest center-to-center distance
  between any two sites (used by the constraint checker).
- ``unit_cell_area()``: area of one primitive (super-)cell (used by
  the fill-fraction metric).

The lattice family is treated as a DISCRETE choice and is not part
of the continuous optimization vector. The factory ``make_lattice``
dispatches to the right implementation given a ``LatticeFamily``
enum value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from lightsail.geometry.base import LatticeFamily


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Lattice(ABC):
    """Abstract 2D lattice of hole center positions."""

    period_nm: float

    @abstractmethod
    def generate_sites(self, extent_nm: float) -> list[tuple[float, float]]:
        """Return (x, y) center positions inside a circular patch of
        diameter ``extent_nm``."""
        ...

    @abstractmethod
    def nearest_neighbor_distance(self) -> float:
        """Smallest center-to-center distance in the lattice."""
        ...

    @abstractmethod
    def unit_cell_area(self) -> float:
        """Area per hole (= primitive cell area / holes per primitive cell)."""
        ...


# ---------------------------------------------------------------------------
# Triangular lattice (one hole per primitive cell, 6 NN)
# ---------------------------------------------------------------------------


@dataclass
class TriangularLattice(Lattice):
    """Simple triangular Bravais lattice.

    Basis vectors:
        a1 = (p, 0)
        a2 = (p/2, p*sqrt(3)/2)
    One hole per primitive cell. Six nearest neighbors at distance p.
    """

    period_nm: float

    def generate_sites(self, extent_nm: float) -> list[tuple[float, float]]:
        p = self.period_nm
        R = 0.5 * extent_nm
        a1 = np.array([p, 0.0])
        a2 = np.array([0.5 * p, 0.5 * np.sqrt(3.0) * p])
        n = int(np.ceil(R / p)) + 2

        sites: list[tuple[float, float]] = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                pos = i * a1 + j * a2
                if np.hypot(pos[0], pos[1]) <= R:
                    sites.append((float(pos[0]), float(pos[1])))
        return sites

    def nearest_neighbor_distance(self) -> float:
        return self.period_nm

    def unit_cell_area(self) -> float:
        return 0.5 * np.sqrt(3.0) * self.period_nm ** 2


# ---------------------------------------------------------------------------
# Hexagonal lattice (honeycomb: two holes per primitive cell)
# ---------------------------------------------------------------------------


@dataclass
class HexagonalLattice(Lattice):
    """Honeycomb lattice with two holes per primitive cell.

    The underlying Bravais lattice is triangular with period
    ``p_bravais = period_nm * sqrt(3)``. Two sublattice sites A and B
    sit at (0, 0) and (0, period_nm). The nearest-neighbor distance
    across sublattices is ``period_nm``.
    """

    period_nm: float

    def generate_sites(self, extent_nm: float) -> list[tuple[float, float]]:
        p = self.period_nm
        p_bravais = p * np.sqrt(3.0)
        R = 0.5 * extent_nm
        a1 = np.array([p_bravais, 0.0])
        a2 = np.array([0.5 * p_bravais, 0.5 * np.sqrt(3.0) * p_bravais])
        # Sublattice offsets (A at origin, B displaced by nearest-neighbor vector)
        offsets = [np.array([0.0, 0.0]), np.array([0.0, p])]

        n = int(np.ceil(R / p_bravais)) + 2
        sites: list[tuple[float, float]] = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                base = i * a1 + j * a2
                for off in offsets:
                    pos = base + off
                    if np.hypot(pos[0], pos[1]) <= R:
                        sites.append((float(pos[0]), float(pos[1])))
        return sites

    def nearest_neighbor_distance(self) -> float:
        return self.period_nm

    def unit_cell_area(self) -> float:
        # Bravais primitive cell area / 2 holes per cell
        p_bravais = self.period_nm * np.sqrt(3.0)
        return 0.5 * (0.5 * np.sqrt(3.0) * p_bravais ** 2)


# ---------------------------------------------------------------------------
# Rectangular lattice (independent x/y periods, one hole per cell)
# ---------------------------------------------------------------------------


@dataclass
class RectangularLattice(Lattice):
    """Orthogonal lattice with independent x and y periods.

    Basis vectors:
        a1 = (period_x, 0)
        a2 = (0,        period_y)

    One hole per primitive cell. Nearest-neighbor distance is
    ``min(period_x, period_y)``.
    """

    period_nm: float              # legacy alias — equals period_x_nm
    period_x_nm: float = 0.0
    period_y_nm: float = 0.0

    def __post_init__(self) -> None:
        # Back-compat: allow ``period_nm=...`` as the x-period when
        # period_x_nm / period_y_nm are not given.
        if self.period_x_nm <= 0:
            self.period_x_nm = self.period_nm
        if self.period_y_nm <= 0:
            self.period_y_nm = self.period_nm
        self.period_nm = self.period_x_nm  # keep base-class alias consistent

    def generate_sites(self, extent_nm: float) -> list[tuple[float, float]]:
        px = self.period_x_nm
        py = self.period_y_nm
        R = 0.5 * extent_nm
        nx = int(np.ceil(R / px)) + 2
        ny = int(np.ceil(R / py)) + 2

        sites: list[tuple[float, float]] = []
        for i in range(-nx, nx + 1):
            for j in range(-ny, ny + 1):
                x = i * px
                y = j * py
                if np.hypot(x, y) <= R:
                    sites.append((float(x), float(y)))
        return sites

    def nearest_neighbor_distance(self) -> float:
        return float(min(self.period_x_nm, self.period_y_nm))

    def unit_cell_area(self) -> float:
        return float(self.period_x_nm * self.period_y_nm)


# ---------------------------------------------------------------------------
# Pentagonal-like supercell
# ---------------------------------------------------------------------------


@dataclass
class PentagonalSupercell(Lattice):
    """Square supercell with 5-fold local hole arrangement.

    True 5-fold periodic lattices do not exist in 2D (crystallographic
    restriction). We approximate a "pentagonal-like" motif by packing
    five holes at the vertices of a regular pentagon inside each
    supercell of a coarse square tiling. One hole may also sit at the
    supercell center (``include_center=True``) to emulate a 1+5 motif.

    - ``period_nm``: side length of the square supercell
    - ``motif_radius_fraction``: fraction of ``period_nm`` at which
      the 5 pentagon vertices are placed (nominal 0.3)
    - ``include_center``: if True, add a 6th hole at the supercell center
    """

    period_nm: float
    motif_radius_fraction: float = 0.32
    include_center: bool = False

    def _motif_offsets(self) -> list[np.ndarray]:
        r = self.motif_radius_fraction * self.period_nm
        offs = []
        for k in range(5):
            angle = 2.0 * np.pi * k / 5.0 + np.pi / 2.0  # point-up pentagon
            offs.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
        if self.include_center:
            offs.append(np.array([0.0, 0.0]))
        return offs

    def generate_sites(self, extent_nm: float) -> list[tuple[float, float]]:
        p = self.period_nm
        R = 0.5 * extent_nm
        offs = self._motif_offsets()
        n = int(np.ceil(R / p)) + 2

        sites: list[tuple[float, float]] = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                center = np.array([i * p, j * p])
                for off in offs:
                    pos = center + off
                    if np.hypot(pos[0], pos[1]) <= R:
                        sites.append((float(pos[0]), float(pos[1])))
        return sites

    def nearest_neighbor_distance(self) -> float:
        """Minimum center-to-center distance within one motif.

        For a regular pentagon with circumscribed radius r, the side
        length (= nearest-neighbor distance) is ``2 * r * sin(pi/5)``.
        If ``include_center`` is True, the center-to-vertex distance
        ``r`` is the tighter bound.
        """
        r = self.motif_radius_fraction * self.period_nm
        side = 2.0 * r * np.sin(np.pi / 5.0)
        if self.include_center:
            return float(min(side, r))
        return float(side)

    def unit_cell_area(self) -> float:
        motif_size = 5 + (1 if self.include_center else 0)
        return (self.period_nm ** 2) / motif_size


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_lattice(
    family: LatticeFamily,
    period_nm: float,
    period_y_nm: float | None = None,
    **kwargs,
) -> Lattice:
    """Factory: return a Lattice instance for the given family.

    ``period_nm`` is the x-period; ``period_y_nm`` is only used by
    :class:`RectangularLattice` (defaults to ``period_nm`` so the
    back-compat "single period" semantics still work for that class).
    """
    if family == LatticeFamily.TRIANGULAR:
        return TriangularLattice(period_nm=period_nm)
    if family == LatticeFamily.HEXAGONAL:
        return HexagonalLattice(period_nm=period_nm)
    if family == LatticeFamily.RECTANGULAR:
        py = period_y_nm if period_y_nm and period_y_nm > 0 else period_nm
        return RectangularLattice(
            period_nm=period_nm,
            period_x_nm=period_nm,
            period_y_nm=py,
        )
    if family == LatticeFamily.PENTAGONAL_SUPERCELL:
        return PentagonalSupercell(period_nm=period_nm, **kwargs)
    if family == LatticeFamily.DUAL_TRIANGULAR:
        # The DUAL_TRIANGULAR supercell is handled by the RCWA solver
        # directly. For site generation and constraint checking we reuse
        # a TriangularLattice with the same period — the two-hole motif
        # is only relevant inside the RCWA unit cell, not for the patch
        # extents / nearest-neighbor computation used by constraints.
        return TriangularLattice(period_nm=period_nm)
    if family == LatticeFamily.DISORDERED_TRIANGULAR:
        # The DISORDERED_TRIANGULAR supercell is handled by the RCWA solver
        # via jittered offsets stored in Structure.metadata. For site
        # generation and constraint checking we fall back to a TriangularLattice
        # with the same period (nearest-neighbor and unit-cell-area are
        # approximate but sufficient for fabrication constraint evaluation).
        return TriangularLattice(period_nm=period_nm)
    raise ValueError(f"Unknown lattice family: {family}")
