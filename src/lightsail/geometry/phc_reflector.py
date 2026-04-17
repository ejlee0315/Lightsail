"""Stage 1 geometry: hole-based photonic crystal reflector.

Continuous design parameters (optimization vector, 7 or 8 dims):
    thickness_nm
    lattice_period_nm           (x-period for rectangular lattice)
    hole_a_rel                  (hole a / period_ref, period_ref = min(px, py))
    hole_b_rel                  (hole b / period_ref)
    hole_rotation_deg
    corner_rounding
    shape_parameter             (rounded to int -> n_sides in [3, 8])
    [lattice_aspect_ratio]      (RECTANGULAR only: period_y / period_x)

Discrete choice (constructor only, NOT part of the vector):
    lattice_family              (TRIANGULAR / HEXAGONAL / RECTANGULAR /
                                 PENTAGONAL_SUPERCELL)

``n_rings`` controls the radial extent of the PhC patch (not the hole
shape) and is also a constructor-level setting, not an optimization
parameter.

Why *relative* hole sizes?
--------------------------
Previous versions had ``hole_a_nm`` and ``hole_b_nm`` as absolute
optimization variables with bounds ``[100, 900]`` nm. That meant the
optimizer could propose combinations like ``(period=400, a=900)`` which
are geometrically impossible (hole larger than the unit cell). Roughly
half of the initial Sobol samples landed in that infeasible region and
BO wasted budget learning to avoid it.

Switching to relative ``a_rel = a/period_ref`` with an upper bound of
``0.48`` makes "hole fits in unit cell" automatic. For rectangular
lattices ``period_ref`` is the smaller of the two periods so the hole
is guaranteed to fit along both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightsail.geometry.base import (
    Hole,
    HoleShape,
    LatticeFamily,
    Material,
    ParametricGeometry,
    Structure,
)
from lightsail.geometry.lattices import Lattice, make_lattice


# Allowed polygon side counts (shape_parameter rounds into this range)
_MIN_N_SIDES = 3
_MAX_N_SIDES = 8

# Upper bound on (a/period_ref, b/period_ref). Just under 0.5 so 2a
# never equals the lattice period — guarantees a nonzero wall before
# the fab check fires.
_HOLE_REL_MAX = 0.48
_HOLE_REL_MIN = 0.05

# Minimum feature/gap constraint used in the frac → rel mapping.
# When hole_a_frac ∈ [0, 1], the actual semi-axis is:
#   a_nm = _HOLE_MIN_RADIUS_NM + frac × ((period-_MIN_WALL_NM)/2 - _HOLE_MIN_RADIUS_NM)
# This guarantees: hole diameter ≥ 2×_HOLE_MIN_RADIUS_NM and
#                  wall ≥ _MIN_WALL_NM for any period in [400, 2000].
_HOLE_MIN_RADIUS_NM = 50.0    # semi-axis min → feature ≥ 100 nm diameter
_MIN_WALL_NM = 100.0           # minimum wall between adjacent holes

# SiN film thickness bounds (nm). Kept at 1000 nm for fabrication
# realism and lightsail momentum-per-mass reasons. An earlier experiment
# on 2026-04-15 relaxed the upper bound to 2500 nm to push MIR
# emissivity past ~0.30; it worked (MIR 0.43 at t=2500) but 2500 nm SiN
# is hard to fab and the extra mass hurts propulsion, so the decision
# was to keep 1000 nm and pursue MIR gains through material/structure
# changes instead. Keep the fabrication constraint ``thickness_range_nm``
# in the config aligned with this.
_THICKNESS_MIN_NM = 200.0
_THICKNESS_MAX_NM = 1000.0

# Rectangular-only: period_y / period_x bounds. Symmetric around 1 on
# log scale so triangular-ish aspect ratios and elongated rectangles
# are equally reachable.
_ASPECT_MIN = 0.5
_ASPECT_MAX = 2.0


@dataclass
class PhCReflector(ParametricGeometry):
    """Central photonic-crystal reflector with rounded-polygon holes.

    The hole shape is a parameterized rounded polygon (see
    :class:`HoleShape`). The lattice is one of a small discrete set
    of families, chosen at construction time.

    Hole sizes are stored relatively (``hole_a_rel = a_nm / period_ref``)
    so the mathematical feasibility of the unit cell is built into the
    parameterization. The absolute ``hole_a_nm`` / ``hole_b_nm`` values
    are still exposed as read-only properties for downstream code.

    For :class:`LatticeFamily.RECTANGULAR` the optimization vector gains
    an extra field ``lattice_aspect_ratio = period_y / period_x``
    (bounds ``[0.5, 2.0]``). For all other families this field is
    fixed to 1.0 and does not appear in the param vector.
    """

    # --- discrete, constructor-level --------------------------------------
    lattice_family: LatticeFamily = LatticeFamily.TRIANGULAR
    n_rings: int = 6  # radial extent control (not optimized)

    # --- continuous optimization variables --------------------------------
    thickness_nm: float = 500.0
    lattice_period_nm: float = 900.0          # period_x for rectangular
    hole_a_rel: float = 0.30                  # derived from frac in from_param_vector
    hole_b_rel: float = 0.30                  # derived from frac in from_param_vector
    hole_a_frac: float = 0.5                  # BO variable ∈ [0,1] → maps to valid a_nm
    hole_b_frac: float = 0.5                  # BO variable ∈ [0,1] → maps to valid b_nm
    hole_rotation_deg: float = 0.0
    corner_rounding: float = 0.5
    shape_parameter: float = 6.0              # rounds to n_sides
    lattice_aspect_ratio: float = 1.0         # period_y / period_x (rect only)

    # --- cached lattice (rebuilt on param update) -------------------------
    _lattice: Lattice = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rebuild_lattice()
        # If user constructed with explicit hole_a_rel/hole_b_rel (legacy or
        # test code), reverse-map them to frac so hole_a_nm stays consistent.
        # Detect this by checking if frac is at default (0.5) but rel is not
        # at the value frac=0.5 would produce.
        expected_rel_a = self._frac_to_radius_nm(self.hole_a_frac) / max(self._period_ref_nm, 1.0)
        if abs(self.hole_a_rel - expected_rel_a) > 1e-6:
            # User set hole_a_rel directly — reverse-map to frac
            a_nm = self.hole_a_rel * self._period_ref_nm
            self.hole_a_frac = self._radius_nm_to_frac(a_nm)
        expected_rel_b = self._frac_to_radius_nm(self.hole_b_frac) / max(self._period_ref_nm, 1.0)
        if abs(self.hole_b_rel - expected_rel_b) > 1e-6:
            b_nm = self.hole_b_rel * self._period_ref_nm
            self.hole_b_frac = self._radius_nm_to_frac(b_nm)
        # Final sync: ensure rel matches frac
        self._sync_rel_from_frac()

    # ------------------------------------------------------------------
    # Derived period / hole nm properties
    # ------------------------------------------------------------------

    @property
    def lattice_period_x_nm(self) -> float:
        return float(self.lattice_period_nm)

    @property
    def lattice_period_y_nm(self) -> float:
        if self.lattice_family == LatticeFamily.RECTANGULAR:
            return float(self.lattice_period_nm * self.lattice_aspect_ratio)
        return float(self.lattice_period_nm)

    @property
    def _period_ref_nm(self) -> float:
        """Reference period for absolute hole sizing.

        For rectangular lattices we use ``min(period_x, period_y)`` so
        holes with ``hole_a_rel ≤ 0.48`` still fit along the shorter
        axis. Other families reduce to ``period_x`` (= period_y).
        """
        if self.lattice_family == LatticeFamily.RECTANGULAR:
            return float(min(self.lattice_period_x_nm, self.lattice_period_y_nm))
        return float(self.lattice_period_nm)

    def _frac_to_radius_nm(self, frac: float) -> float:
        """Map frac ∈ [0, 1] to hole semi-axis in nm, guaranteed feasible.

        a_nm = _HOLE_MIN_RADIUS_NM + frac × ((period_ref - _MIN_WALL_NM)/2 - _HOLE_MIN_RADIUS_NM)
        Ensures: diameter ≥ 2×_HOLE_MIN_RADIUS_NM and wall ≥ _MIN_WALL_NM.
        """
        p = self._period_ref_nm
        a_max = (p - _MIN_WALL_NM) / 2.0
        a_min = _HOLE_MIN_RADIUS_NM
        if a_max <= a_min:
            return a_min
        return float(a_min + frac * (a_max - a_min))

    def _radius_nm_to_frac(self, a_nm: float) -> float:
        """Inverse of _frac_to_radius_nm."""
        p = self._period_ref_nm
        a_max = (p - _MIN_WALL_NM) / 2.0
        a_min = _HOLE_MIN_RADIUS_NM
        if a_max <= a_min:
            return 0.5
        return float(np.clip((a_nm - a_min) / (a_max - a_min), 0.0, 1.0))

    def _sync_rel_from_frac(self) -> None:
        """Update hole_a_rel / hole_b_rel from the current frac values."""
        p = self._period_ref_nm
        if p > 0:
            self.hole_a_rel = self._frac_to_radius_nm(self.hole_a_frac) / p
            self.hole_b_rel = self._frac_to_radius_nm(self.hole_b_frac) / p

    @property
    def hole_a_nm(self) -> float:
        """Absolute ``a`` semi-axis in nm.

        Uses ``hole_a_rel × period_ref``. The rel value is kept in sync
        with hole_a_frac by __post_init__ and from_param_vector, but
        can also be set directly for backward compatibility.
        """
        return float(self.hole_a_rel * self._period_ref_nm)

    @property
    def hole_b_nm(self) -> float:
        """Absolute ``b`` semi-axis in nm (``b_rel × period_ref``)."""
        return float(self.hole_b_rel * self._period_ref_nm)

    # ------------------------------------------------------------------
    # ParametricGeometry interface
    # ------------------------------------------------------------------

    def _uses_aspect_ratio(self) -> bool:
        return self.lattice_family == LatticeFamily.RECTANGULAR

    def param_names(self) -> list[str]:
        base = [
            "thickness_nm",
            "lattice_period_nm",
            "hole_a_frac",
            "hole_b_frac",
            "hole_rotation_deg",
            "corner_rounding",
            "shape_parameter",
        ]
        if self._uses_aspect_ratio():
            return base + ["lattice_aspect_ratio"]
        return base

    def param_bounds(self) -> list[tuple[float, float]]:
        base = [
            (_THICKNESS_MIN_NM, _THICKNESS_MAX_NM),  # thickness_nm
            (400.0, 2000.0),                 # lattice_period_nm (period_x)
            (0.0, 1.0),                      # hole_a_frac → maps to valid a_nm
            (0.0, 1.0),                      # hole_b_frac → maps to valid b_nm
            (0.0, 180.0),                    # hole_rotation_deg
            (0.0, 1.0),                      # corner_rounding
            (float(_MIN_N_SIDES), float(_MAX_N_SIDES)),  # shape_parameter
        ]
        if self._uses_aspect_ratio():
            base.append((_ASPECT_MIN, _ASPECT_MAX))
        return base

    def to_param_vector(self) -> np.ndarray:
        values = [
            self.thickness_nm,
            self.lattice_period_nm,
            self.hole_a_frac,
            self.hole_b_frac,
            self.hole_rotation_deg,
            self.corner_rounding,
            self.shape_parameter,
        ]
        if self._uses_aspect_ratio():
            values.append(self.lattice_aspect_ratio)
        return np.array(values, dtype=float)

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        expected = 8 if self._uses_aspect_ratio() else 7
        if v.size != expected:
            raise ValueError(
                f"PhCReflector({self.lattice_family.value}) expects "
                f"{expected} params, got {v.size}"
            )
        self.thickness_nm = float(v[0])
        self.lattice_period_nm = float(v[1])
        self.hole_a_frac = float(np.clip(v[2], 0.0, 1.0))
        self.hole_b_frac = float(np.clip(v[3], 0.0, 1.0))
        self.hole_rotation_deg = float(v[4])
        self.corner_rounding = float(np.clip(v[5], 0.0, 1.0))
        self.shape_parameter = float(v[6])
        if self._uses_aspect_ratio():
            self.lattice_aspect_ratio = float(
                np.clip(v[7], _ASPECT_MIN, _ASPECT_MAX)
            )
        else:
            self.lattice_aspect_ratio = 1.0
        # Sync rel values from frac + period
        self._sync_rel_from_frac()
        self._rebuild_lattice()

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------

    @property
    def n_sides(self) -> int:
        """Integer polygon sides derived from ``shape_parameter``."""
        n = int(round(self.shape_parameter))
        return max(_MIN_N_SIDES, min(_MAX_N_SIDES, n))

    def hole_shape(self) -> HoleShape:
        """Return the HoleShape implied by the current parameters."""
        return HoleShape(
            a_nm=self.hole_a_nm,
            b_nm=self.hole_b_nm,
            n_sides=self.n_sides,
            rotation_deg=self.hole_rotation_deg,
            corner_rounding=self.corner_rounding,
        )

    def _rebuild_lattice(self) -> None:
        if self.lattice_family == LatticeFamily.RECTANGULAR:
            self._lattice = make_lattice(
                self.lattice_family,
                period_nm=self.lattice_period_x_nm,
                period_y_nm=self.lattice_period_y_nm,
            )
        else:
            self._lattice = make_lattice(
                self.lattice_family, self.lattice_period_nm
            )

    def generate_holes(self) -> list[Hole]:
        """Place the current hole shape at every lattice site."""
        shape = self.hole_shape()
        extent = 2.0 * self.outer_radius_nm
        sites = self._lattice.generate_sites(extent)
        return [Hole(x_nm=x, y_nm=y, shape=shape) for (x, y) in sites]

    def to_structure(self) -> Structure:
        holes = self.generate_holes()
        return Structure(
            material=Material.SIN,
            thickness_nm=self.thickness_nm,
            lattice_family=self.lattice_family,
            lattice_period_nm=self.lattice_period_nm,
            period_x_nm=self.lattice_period_x_nm,
            period_y_nm=self.lattice_period_y_nm,
            holes=holes,
            extent_nm=2.0 * self.outer_radius_nm,
            metadata={
                "lattice_family": self.lattice_family.value,
                "n_sides": self.n_sides,
                "n_holes": len(holes),
                "nearest_neighbor_nm": self.nearest_neighbor_distance_nm,
                "unit_cell_area_nm2": self.unit_cell_area_nm2,
                "hole_a_rel": self.hole_a_rel,
                "hole_b_rel": self.hole_b_rel,
                "lattice_period_x_nm": self.lattice_period_x_nm,
                "lattice_period_y_nm": self.lattice_period_y_nm,
                "lattice_aspect_ratio": self.lattice_aspect_ratio,
            },
        )

    # ------------------------------------------------------------------
    # Derived quantities (used by pipeline and constraint checker)
    # ------------------------------------------------------------------

    @property
    def outer_radius_nm(self) -> float:
        """Outer radius of the PhC patch (uses the larger period)."""
        return float(self.n_rings * max(self.lattice_period_x_nm, self.lattice_period_y_nm))

    @property
    def nearest_neighbor_distance_nm(self) -> float:
        return float(self._lattice.nearest_neighbor_distance())

    @property
    def unit_cell_area_nm2(self) -> float:
        return float(self._lattice.unit_cell_area())

    @property
    def lattice(self) -> Lattice:
        """Read-only access to the internal Lattice instance."""
        return self._lattice


# ======================================================================
# Freeform extension — Fourier boundary modulation
# ======================================================================

# Amplitude bounds for each Fourier harmonic (radial modulation depth).
# 0.25 keeps holes well-behaved (no self-intersection).
_FOURIER_AMP_MAX = 0.25


@dataclass
class FreeformPhCReflector(PhCReflector):
    """PhCReflector with Fourier boundary harmonics on the hole shape.

    Adds 4 continuous optimization parameters on top of the base 7:

    - ``fourier_amp2``: amplitude of the n=2 radial harmonic [0, 0.25]
    - ``fourier_phase2``: phase of the n=2 harmonic [0, 2π]
    - ``fourier_amp3``: amplitude of the n=3 radial harmonic [0, 0.25]
    - ``fourier_phase3``: phase of the n=3 harmonic [0, 2π]

    The hole boundary becomes::

        r(θ) = r_base(θ) × (1 + amp2·cos(2θ + φ2) + amp3·cos(3θ + φ3))

    where ``r_base`` is the existing polygon/ellipse blend from
    :class:`PhCReflector`.

    n=2 creates butterfly/peanut shapes (quadrupolar distortion).
    n=3 creates trefoil shapes (can match triangular lattice symmetry).
    Higher harmonics (n=4, 5) can be added later by increasing
    ``n_harmonics`` at construction time.

    Total parameter count: 11 for triangular (7 base + 4 Fourier),
    12 for rectangular (8 base + 4 Fourier).
    """

    fourier_amp2: float = 0.0
    fourier_phase2: float = 0.0
    fourier_amp3: float = 0.0
    fourier_phase3: float = 0.0

    # ------------------------------------------------------------------
    # ParametricGeometry interface overrides
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        return super().param_names() + [
            "fourier_amp2",
            "fourier_phase2",
            "fourier_amp3",
            "fourier_phase3",
        ]

    def param_bounds(self) -> list[tuple[float, float]]:
        return super().param_bounds() + [
            (0.0, _FOURIER_AMP_MAX),       # fourier_amp2
            (0.0, 2.0 * np.pi),            # fourier_phase2
            (0.0, _FOURIER_AMP_MAX),       # fourier_amp3
            (0.0, 2.0 * np.pi),            # fourier_phase3
        ]

    def to_param_vector(self) -> np.ndarray:
        base = super().to_param_vector()
        return np.append(base, [
            self.fourier_amp2,
            self.fourier_phase2,
            self.fourier_amp3,
            self.fourier_phase3,
        ])

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        # Last 4 entries are Fourier params; pass the rest to parent
        super().from_param_vector(v[:-4])
        self.fourier_amp2 = float(np.clip(v[-4], 0.0, _FOURIER_AMP_MAX))
        self.fourier_phase2 = float(v[-3] % (2.0 * np.pi))
        self.fourier_amp3 = float(np.clip(v[-2], 0.0, _FOURIER_AMP_MAX))
        self.fourier_phase3 = float(v[-1] % (2.0 * np.pi))

    # ------------------------------------------------------------------
    # Override hole_shape to include Fourier modulation
    # ------------------------------------------------------------------

    def hole_shape(self) -> HoleShape:
        return HoleShape(
            a_nm=self.hole_a_nm,
            b_nm=self.hole_b_nm,
            n_sides=self.n_sides,
            rotation_deg=self.hole_rotation_deg,
            corner_rounding=self.corner_rounding,
            fourier_amplitudes=(self.fourier_amp2, self.fourier_amp3),
            fourier_phases=(self.fourier_phase2, self.fourier_phase3),
        )


# ======================================================================
# Dual-hole supercell extension — two distinct hole sizes per unit cell
# ======================================================================


@dataclass
class DualHolePhCReflector(PhCReflector):
    """PhCReflector with a dual-hole supercell (DUAL_TRIANGULAR lattice).

    Places two holes of *different* sizes at the two offset positions of
    a rectangular supercell that tiles the triangular Bravais lattice.
    The supercell vectors are::

        L1 = (period, 0)
        L2 = (0, period * sqrt(3))

    with hole 1 at (0, 0) and hole 2 at (period/2, period*sqrt(3)/2).

    The two holes share the same polygon type (n_sides, rotation,
    rounding) but have independent relative radii, allowing the optimizer
    to create two distinct resonance peaks and thereby broaden the
    reflectivity plateau.

    Adds 2 extra continuous optimization parameters on top of the base 7:

    - ``hole_a_rel_2``: a / period_ref for the second hole [0.05, 0.48]
    - ``hole_b_rel_2``: b / period_ref for the second hole [0.05, 0.48]

    The ``lattice_family`` is fixed to :attr:`LatticeFamily.DUAL_TRIANGULAR`
    at construction time (cannot be changed).

    Total parameter count: 9.
    """

    hole_a_rel_2: float = 0.30
    hole_b_rel_2: float = 0.30

    def __post_init__(self) -> None:
        # Force the lattice family — users should not change it.
        self.lattice_family = LatticeFamily.DUAL_TRIANGULAR
        super().__post_init__()

    # ------------------------------------------------------------------
    # Rebuild lattice: DUAL_TRIANGULAR reuses TriangularLattice for site
    # generation (visualization only). RCWA uses the supercell directly.
    # ------------------------------------------------------------------

    def _rebuild_lattice(self) -> None:
        # DUAL_TRIANGULAR is not handled by make_lattice; use the
        # TriangularLattice with the same period as a best-effort
        # approximation for site generation (visualization / constraints).
        from lightsail.geometry.lattices import TriangularLattice

        self._lattice = TriangularLattice(period_nm=self.lattice_period_nm)

    # ------------------------------------------------------------------
    # ParametricGeometry interface overrides
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        # Base class returns the 7 standard names (no aspect ratio for
        # DUAL_TRIANGULAR since the supercell shape is fixed by the
        # triangular geometry).
        base = [
            "thickness_nm",
            "lattice_period_nm",
            "hole_a_rel",
            "hole_b_rel",
            "hole_rotation_deg",
            "corner_rounding",
            "shape_parameter",
        ]
        return base + ["hole_a_rel_2", "hole_b_rel_2"]

    def param_bounds(self) -> list[tuple[float, float]]:
        base = [
            (_THICKNESS_MIN_NM, _THICKNESS_MAX_NM),
            (400.0, 2000.0),
            (_HOLE_REL_MIN, _HOLE_REL_MAX),
            (_HOLE_REL_MIN, _HOLE_REL_MAX),
            (0.0, 180.0),
            (0.0, 1.0),
            (float(_MIN_N_SIDES), float(_MAX_N_SIDES)),
        ]
        return base + [
            (_HOLE_REL_MIN, _HOLE_REL_MAX),  # hole_a_rel_2
            (_HOLE_REL_MIN, _HOLE_REL_MAX),  # hole_b_rel_2
        ]

    def to_param_vector(self) -> np.ndarray:
        return np.array(
            [
                self.thickness_nm,
                self.lattice_period_nm,
                self.hole_a_rel,
                self.hole_b_rel,
                self.hole_rotation_deg,
                self.corner_rounding,
                self.shape_parameter,
                self.hole_a_rel_2,
                self.hole_b_rel_2,
            ],
            dtype=float,
        )

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        if v.size != 9:
            raise ValueError(
                f"DualHolePhCReflector expects 9 params, got {v.size}"
            )
        self.thickness_nm = float(v[0])
        self.lattice_period_nm = float(v[1])
        self.hole_a_rel = float(np.clip(v[2], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self.hole_b_rel = float(np.clip(v[3], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self.hole_rotation_deg = float(v[4])
        self.corner_rounding = float(np.clip(v[5], 0.0, 1.0))
        self.shape_parameter = float(v[6])
        self.hole_a_rel_2 = float(np.clip(v[7], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self.hole_b_rel_2 = float(np.clip(v[8], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self._rebuild_lattice()

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------

    def hole_shape_2(self) -> HoleShape:
        """Return the HoleShape for the second hole in the supercell."""
        return HoleShape(
            a_nm=self.hole_a_rel_2 * self.lattice_period_nm,
            b_nm=self.hole_b_rel_2 * self.lattice_period_nm,
            n_sides=self.n_sides,
            rotation_deg=self.hole_rotation_deg,
            corner_rounding=self.corner_rounding,
        )

    def generate_holes(self) -> list[Hole]:
        """Place the two hole shapes at the two supercell offset positions.

        For visualization / constraint checking only. RCWA uses the
        offsets stored inside the unit cell, not these Hole objects
        directly.
        """
        shape1 = self.hole_shape()
        shape2 = self.hole_shape_2()
        p = self.lattice_period_nm
        sqrt3 = np.sqrt(3.0)
        # Two canonical offset positions of the rectangular supercell.
        offsets = [(0.0, 0.0), (p * 0.5, p * sqrt3 * 0.5)]
        holes = []
        for ox, oy in offsets:
            shape = shape1 if ox == 0.0 else shape2
            holes.append(Hole(x_nm=ox, y_nm=oy, shape=shape))
        return holes

    def to_structure(self) -> Structure:
        shape1 = self.hole_shape()
        shape2 = self.hole_shape_2()
        holes = self.generate_holes()
        p = self.lattice_period_nm
        sqrt3 = np.sqrt(3.0)
        return Structure(
            material=Material.SIN,
            thickness_nm=self.thickness_nm,
            lattice_family=LatticeFamily.DUAL_TRIANGULAR,
            lattice_period_nm=p,
            period_x_nm=p,
            period_y_nm=p * sqrt3,
            holes=holes,
            extent_nm=2.0 * self.outer_radius_nm,
            metadata={
                "lattice_family": LatticeFamily.DUAL_TRIANGULAR.value,
                "n_sides": self.n_sides,
                "n_holes": len(holes),
                "nearest_neighbor_nm": self.nearest_neighbor_distance_nm,
                "unit_cell_area_nm2": self.unit_cell_area_nm2,
                "hole_a_rel": self.hole_a_rel,
                "hole_b_rel": self.hole_b_rel,
                "hole_a_rel_2": self.hole_a_rel_2,
                "hole_b_rel_2": self.hole_b_rel_2,
                "lattice_period_x_nm": p,
                "lattice_period_y_nm": p * sqrt3,
                "lattice_aspect_ratio": sqrt3,
                # Per-hole shapes for the RCWA rasterizer.
                "per_hole_shapes": [shape1, shape2],
            },
        )


# ======================================================================
# Disordered lattice extension — positionally jittered 2×2 supercell
# ======================================================================

# Maximum jitter amplitude as a fraction of the lattice period.
_JITTER_AMP_MAX = 0.20


@dataclass
class DisorderedPhCReflector(PhCReflector):
    """PhCReflector with a disordered 2×2 triangular supercell.

    Extends :class:`PhCReflector` with one extra optimization parameter,
    ``jitter_amplitude``, that controls how far each of the 4 holes in the
    2×2 supercell is displaced from its ideal triangular lattice site.

    The 2×2 supercell vectors are::

        L1 = (2*period, 0)
        L2 = (period,   period*sqrt(3))

    which tiles the plane with the same density as the triangular Bravais
    lattice and contains exactly 4 holes. The 4 base positions inside the
    supercell are::

        (0,             0)
        (period,        0)
        (period/2,      period*sqrt(3)/2)
        (3*period/2,    period*sqrt(3)/2)

    Each position is perturbed by ``jitter_amplitude * period * delta_i``
    where ``delta_i`` is a **fixed** unit vector drawn from a seeded RNG
    (seed 12345). The same ``jitter_amplitude`` always produces the same
    structure, making results reproducible and gradients meaningful.

    All 4 holes share a single :class:`HoleShape` (same as the base
    :class:`PhCReflector`). The jitter only displaces positions.

    Total parameter count: 8 (7 base + jitter_amplitude).
    The ``lattice_family`` is fixed to
    :attr:`LatticeFamily.DISORDERED_TRIANGULAR` at construction time.
    """

    jitter_amplitude: float = 0.0

    # Pre-computed unit vectors drawn from seed 12345 (4 holes × 2 coords).
    # Generated once at import time so they are always identical.
    _JITTER_DELTAS: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Force the lattice family — users should not change it.
        self.lattice_family = LatticeFamily.DISORDERED_TRIANGULAR
        # Pre-compute fixed random unit vectors for the 4 supercell holes.
        rng = np.random.RandomState(12345)
        raw = rng.randn(4, 2)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self._JITTER_DELTAS = (raw / norms).astype(float)
        super().__post_init__()

    # ------------------------------------------------------------------
    # Rebuild lattice: reuse TriangularLattice for site generation /
    # constraint checking (same reasoning as DUAL_TRIANGULAR).
    # ------------------------------------------------------------------

    def _rebuild_lattice(self) -> None:
        from lightsail.geometry.lattices import TriangularLattice

        self._lattice = TriangularLattice(period_nm=self.lattice_period_nm)

    # ------------------------------------------------------------------
    # ParametricGeometry interface overrides
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        # Explicit base-7 list (no aspect ratio for fixed supercell).
        base = [
            "thickness_nm",
            "lattice_period_nm",
            "hole_a_rel",
            "hole_b_rel",
            "hole_rotation_deg",
            "corner_rounding",
            "shape_parameter",
        ]
        return base + ["jitter_amplitude"]

    def param_bounds(self) -> list[tuple[float, float]]:
        base = [
            (_THICKNESS_MIN_NM, _THICKNESS_MAX_NM),
            (400.0, 2000.0),
            (_HOLE_REL_MIN, _HOLE_REL_MAX),
            (_HOLE_REL_MIN, _HOLE_REL_MAX),
            (0.0, 180.0),
            (0.0, 1.0),
            (float(_MIN_N_SIDES), float(_MAX_N_SIDES)),
        ]
        return base + [(0.0, _JITTER_AMP_MAX)]

    def to_param_vector(self) -> np.ndarray:
        return np.array(
            [
                self.thickness_nm,
                self.lattice_period_nm,
                self.hole_a_rel,
                self.hole_b_rel,
                self.hole_rotation_deg,
                self.corner_rounding,
                self.shape_parameter,
                self.jitter_amplitude,
            ],
            dtype=float,
        )

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        if v.size != 8:
            raise ValueError(
                f"DisorderedPhCReflector expects 8 params, got {v.size}"
            )
        self.thickness_nm = float(v[0])
        self.lattice_period_nm = float(v[1])
        self.hole_a_rel = float(np.clip(v[2], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self.hole_b_rel = float(np.clip(v[3], _HOLE_REL_MIN, _HOLE_REL_MAX))
        self.hole_rotation_deg = float(v[4])
        self.corner_rounding = float(np.clip(v[5], 0.0, 1.0))
        self.shape_parameter = float(v[6])
        self.jitter_amplitude = float(np.clip(v[7], 0.0, _JITTER_AMP_MAX))
        self._rebuild_lattice()

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------

    def _disordered_offsets(self) -> list[tuple[float, float]]:
        """Compute the 4 jittered hole positions inside the 2×2 supercell.

        Base positions are the ideal triangular lattice sites inside the
        2×2 supercell. Each position is displaced by
        ``jitter_amplitude * period * delta_i`` where ``delta_i`` is a
        fixed unit vector (seeded RNG, seed 12345).
        """
        p = self.lattice_period_nm
        sqrt3 = np.sqrt(3.0)
        base = [
            (0.0,           0.0),
            (p,             0.0),
            (p * 0.5,       p * sqrt3 * 0.5),
            (p * 1.5,       p * sqrt3 * 0.5),
        ]
        jitter = self.jitter_amplitude * p
        offsets = []
        for i, (bx, by) in enumerate(base):
            dx, dy = self._JITTER_DELTAS[i]
            offsets.append((bx + jitter * dx, by + jitter * dy))
        return offsets

    def generate_holes(self) -> list[Hole]:
        """Place the shared hole shape at jittered supercell positions."""
        shape = self.hole_shape()
        offsets = self._disordered_offsets()
        return [Hole(x_nm=x, y_nm=y, shape=shape) for (x, y) in offsets]

    def to_structure(self) -> Structure:
        p = self.lattice_period_nm
        sqrt3 = np.sqrt(3.0)
        offsets = self._disordered_offsets()
        holes = [Hole(x_nm=x, y_nm=y, shape=self.hole_shape()) for (x, y) in offsets]
        return Structure(
            material=Material.SIN,
            thickness_nm=self.thickness_nm,
            lattice_family=LatticeFamily.DISORDERED_TRIANGULAR,
            lattice_period_nm=p,
            # Supercell dimensions: L1 = (2p, 0), L2 = (p, p*sqrt(3)).
            period_x_nm=2.0 * p,
            period_y_nm=p * sqrt3,
            holes=holes,
            extent_nm=2.0 * self.outer_radius_nm,
            metadata={
                "lattice_family": LatticeFamily.DISORDERED_TRIANGULAR.value,
                "n_sides": self.n_sides,
                "n_holes": len(holes),
                "nearest_neighbor_nm": self.nearest_neighbor_distance_nm,
                "unit_cell_area_nm2": self.unit_cell_area_nm2,
                "hole_a_rel": self.hole_a_rel,
                "hole_b_rel": self.hole_b_rel,
                "jitter_amplitude": self.jitter_amplitude,
                "lattice_period_x_nm": 2.0 * p,
                "lattice_period_y_nm": p * sqrt3,
                # 4 jittered hole positions for the RCWA unit cell.
                "disordered_offsets_nm": tuple(
                    (float(x), float(y)) for (x, y) in offsets
                ),
            },
        )
