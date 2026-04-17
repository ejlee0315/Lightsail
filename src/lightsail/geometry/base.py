"""Base classes and core data types for parametric geometry representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Material(Enum):
    """Supported membrane materials."""

    SIN = "SiN"


class LatticeFamily(Enum):
    """Discrete lattice family choice for the central PhC reflector.

    Treated as a discrete/categorical parameter — NOT a continuous
    optimization variable. The optimizer runs separately per family
    and results are compared afterwards.

    Notes:
      - TRIANGULAR: 60° parallelogram primitive cell, isotropic periodicity.
      - HEXAGONAL: honeycomb (two sublattice sites per primitive cell).
      - RECTANGULAR: orthogonal lattice with independent x/y periods. Adds
        one extra continuous optimization variable (``lattice_aspect_ratio``
        = ``period_y / period_x``).
      - PENTAGONAL_SUPERCELL: square supercell with a 5-vertex motif.
    """

    TRIANGULAR = "triangular"
    HEXAGONAL = "hexagonal"
    RECTANGULAR = "rectangular"
    PENTAGONAL_SUPERCELL = "pentagonal_supercell"
    DUAL_TRIANGULAR = "dual_triangular"
    DISORDERED_TRIANGULAR = "disordered_triangular"


# ---------------------------------------------------------------------------
# Hole shape (rounded polygon family)
# ---------------------------------------------------------------------------


@dataclass
class HoleShape:
    """Parameterized rounded-polygon hole shape with optional Fourier modes.

    Defined by:
      - a_nm, b_nm: semi-axes of the enclosing ellipse (anisotropy)
      - n_sides: number of polygon sides (3, 4, 5, 6, ...)
      - rotation_deg: in-plane rotation of the shape
      - corner_rounding: in [0, 1]; 0 = sharp n-gon, 1 = ellipse
      - fourier_amplitudes: optional radial modulation amplitudes for
        harmonics n=2, 3, ... (e.g., (0.1, 0.05) for 2nd and 3rd order).
        Each amplitude is in [0, 0.3]; the radius is modulated as
        ``r *= (1 + sum(amp_k * cos(k*psi + phase_k)))``.
      - fourier_phases: phases (rad) for the corresponding harmonics.

    The boundary is generated analytically by linearly blending a
    sharp regular polygon with the inscribing ellipse, then applying
    the Fourier radial modulation.
    """

    a_nm: float
    b_nm: float
    n_sides: int
    rotation_deg: float = 0.0
    corner_rounding: float = 0.5
    fourier_amplitudes: tuple = ()   # (amp2, amp3, ...) for harmonics n=2,3,...
    fourier_phases: tuple = ()       # (phase2, phase3, ...) in radians

    def __post_init__(self) -> None:
        if self.n_sides < 3:
            raise ValueError(f"n_sides must be >= 3, got {self.n_sides}")
        if self.a_nm <= 0 or self.b_nm <= 0:
            raise ValueError("a_nm and b_nm must be positive")
        self.corner_rounding = float(np.clip(self.corner_rounding, 0.0, 1.0))
        self.fourier_amplitudes = tuple(float(a) for a in self.fourier_amplitudes)
        self.fourier_phases = tuple(float(p) for p in self.fourier_phases)

    # -- boundary generation ------------------------------------------------

    def boundary(self, n_pts: int = 256) -> np.ndarray:
        """Return an (n_pts, 2) array of boundary points in local frame.

        Points are centered at the origin; apply the hole's (x, y)
        to translate into the structure's global frame.
        """
        psi = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

        # Sharp n-gon radius in polar form (inscribed in unit circle):
        # for each angle, find its offset from the nearest edge midpoint
        # and use the apothem formula r = cos(pi/n) / cos(phi_rel).
        sector = 2.0 * np.pi / self.n_sides
        phi_rel = np.mod(psi + sector / 2.0, sector) - sector / 2.0
        r_polygon = np.cos(np.pi / self.n_sides) / np.cos(phi_rel)

        # Blend with unit circle based on corner_rounding.
        r_shape = (1.0 - self.corner_rounding) * r_polygon + self.corner_rounding * 1.0

        # Fourier radial modulation: r *= (1 + sum amp_k cos(k*psi + phase_k))
        if self.fourier_amplitudes:
            modulation = np.ones_like(psi)
            for i, (amp, phase) in enumerate(
                zip(self.fourier_amplitudes, self.fourier_phases)
            ):
                k = i + 2  # harmonics start at n=2
                modulation += amp * np.cos(k * psi + phase)
            # Clamp to prevent self-intersection (negative radius)
            modulation = np.clip(modulation, 0.3, 1.7)
            r_shape = r_shape * modulation

        # Place into local-frame ellipse (a, b), then rotate.
        x_loc = self.a_nm * r_shape * np.cos(psi)
        y_loc = self.b_nm * r_shape * np.sin(psi)

        rot = np.radians(self.rotation_deg)
        c, s = np.cos(rot), np.sin(rot)
        x = c * x_loc - s * y_loc
        y = s * x_loc + c * y_loc

        return np.column_stack([x, y])

    # -- geometric metrics --------------------------------------------------

    def min_feature_nm(self, n_pts: int = 256) -> float:
        """Minimum width of the hole (smallest diameter across the shape).

        Computed as twice the smallest distance from center to boundary.
        """
        pts = self.boundary(n_pts)
        radii = np.linalg.norm(pts, axis=1)
        return float(2.0 * radii.min())

    def max_extent_nm(self, n_pts: int = 256) -> float:
        """Maximum hole extent (largest diameter across the shape)."""
        pts = self.boundary(n_pts)
        radii = np.linalg.norm(pts, axis=1)
        return float(2.0 * radii.max())

    def bounding_box_nm(self, n_pts: int = 256) -> tuple[float, float]:
        """Axis-aligned bounding box (width, height) in nm."""
        pts = self.boundary(n_pts)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        return w, h

    def area_nm2(self, n_pts: int = 256) -> float:
        """Enclosed area via the shoelace formula."""
        pts = self.boundary(n_pts)
        x, y = pts[:, 0], pts[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Structure primitives
# ---------------------------------------------------------------------------


@dataclass
class Hole:
    """A single hole placed on the membrane.

    Combines a position with a shared HoleShape. Typically every hole
    in a lattice instance carries the same HoleShape object, so a
    lattice of N holes only uses O(1) extra memory for the shape.
    """

    x_nm: float
    y_nm: float
    shape: HoleShape

    def boundary_global(self, n_pts: int = 128) -> np.ndarray:
        """Boundary points in the structure's global frame."""
        local = self.shape.boundary(n_pts)
        return local + np.array([self.x_nm, self.y_nm])


@dataclass
class Ring:
    """A single concentric ring of the outer metagrating.

    Stores enough information for both visualization and
    constraint checking. Curvature and asymmetry perturb the
    nominal circular boundary.
    """

    inner_radius_nm: float
    outer_radius_nm: float
    curvature: float = 0.0  # radial warping amplitude (1st-order)
    asymmetry: float = 0.0  # angular asymmetry amplitude (2nd-order)

    @property
    def width_nm(self) -> float:
        return self.outer_radius_nm - self.inner_radius_nm

    def boundary(self, n_pts: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Return (inner_boundary, outer_boundary) point arrays.

        Applies curvature (1-fold radial modulation) and asymmetry
        (2-fold radial modulation) to the nominal circle.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        modulation = (
            1.0
            + self.curvature * np.cos(theta)
            + self.asymmetry * np.cos(2.0 * theta)
        )

        r_in = self.inner_radius_nm * modulation
        r_out = self.outer_radius_nm * modulation

        inner = np.column_stack([r_in * np.cos(theta), r_in * np.sin(theta)])
        outer = np.column_stack([r_out * np.cos(theta), r_out * np.sin(theta)])
        return inner, outer


@dataclass
class Structure:
    """Complete structure description ready for simulation.

    This is the interface object passed to the Simulator. It contains
    all geometric and material information needed to set up a run.
    """

    material: Material = Material.SIN
    thickness_nm: float = 500.0
    lattice_family: Optional[LatticeFamily] = None
    lattice_period_nm: Optional[float] = None
    period_x_nm: Optional[float] = None  # rectangular unit-cell period (for RCWA)
    period_y_nm: Optional[float] = None
    holes: list[Hole] = field(default_factory=list)
    rings: list[Ring] = field(default_factory=list)
    extent_nm: Optional[float] = None  # total structure diameter
    metadata: dict = field(default_factory=dict)

    @property
    def has_phc(self) -> bool:
        return len(self.holes) > 0

    @property
    def has_metagrating(self) -> bool:
        return len(self.rings) > 0

    @property
    def total_hole_area_nm2(self) -> float:
        """Sum of enclosed hole areas (for fill-fraction calculation)."""
        return sum(h.shape.area_nm2() for h in self.holes)


# ---------------------------------------------------------------------------
# Parametric geometry interface
# ---------------------------------------------------------------------------


class ParametricGeometry(ABC):
    """Abstract base for low-dimensional parametric geometry.

    Each subclass defines a small number of continuous design
    parameters that fully describe a geometry component (PhC
    reflector or metagrating). The optimizer works in this
    parameter space. Discrete choices such as lattice_family are
    passed via the constructor and are NOT part of the parameter
    vector.
    """

    @abstractmethod
    def param_names(self) -> list[str]:
        """Return ordered list of parameter names."""
        ...

    @abstractmethod
    def param_bounds(self) -> list[tuple[float, float]]:
        """Return (min, max) bounds for each parameter."""
        ...

    @abstractmethod
    def to_param_vector(self) -> np.ndarray:
        """Export current parameters as a flat numpy array."""
        ...

    @abstractmethod
    def from_param_vector(self, vector: np.ndarray) -> None:
        """Update internal state from a parameter vector (in-place)."""
        ...

    @abstractmethod
    def to_structure(self) -> Structure:
        """Convert current parameters into a Structure for simulation."""
        ...

    @property
    def n_params(self) -> int:
        return len(self.param_names())

    def validate_vector(self, vector: np.ndarray) -> bool:
        """Check that a parameter vector is within bounds."""
        bounds = self.param_bounds()
        if len(vector) != len(bounds):
            return False
        for val, (lo, hi) in zip(vector, bounds):
            if val < lo or val > hi:
                return False
        return True
