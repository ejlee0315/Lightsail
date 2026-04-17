"""grcwa-backed RCWA solver conforming to :class:`ElectromagneticSolver`.

This wraps `grcwa <https://github.com/weiliangjinca/grcwa>`_ (Jin & Rodriguez,
pure-NumPy RCWA with autograd) so the rest of the lightsail pipeline can stay
decoupled from the specific RCWA backend.

Design decisions
----------------
* **Units inside grcwa**: all lengths in micrometers, frequency = 1/λ[µm].
  The conversion from the project's nanometer convention happens inside this
  module only.
* **Unit cell**: derived from the :class:`LatticeFamily` of the Structure:
    - triangular → 60° parallelogram primitive cell, one hole per cell
    - hexagonal (honeycomb) → larger primitive cell with two sublattice holes
    - pentagonal_supercell → square supercell holding a 5-hole motif
    - None (slab / metagrating) → square 1 µm dummy cell
  The hole shapes are rasterized onto an ``Nx × Ny`` grid of relative
  permittivities.
* **Dispersion**: the :class:`SiNDispersion` model provides the complex ε at
  each wavelength, so the solver handles NIR (Luke) and MIR (Kischkat)
  automatically.
* **Polarization**: results are averaged over TE/TM so the objective
  functions see an unpolarized response (matches how the previous MockSolver
  behaved).
* **Stage 2 metagrating**: the rings are not periodic, so this solver falls
  back to a plain SiN slab for metagrating-only structures. Stabilization is
  still evaluated via the analytic :class:`AsymmetryStabilizationProxy` as
  requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from lightsail.geometry.base import (
    HoleShape,
    LatticeFamily,
    Structure,
)
from lightsail.materials import SiNDispersion
from lightsail.simulation.base import ElectromagneticSolver

logger = logging.getLogger(__name__)


@dataclass
class RCWAConfig:
    """Runtime knobs for :class:`RCWASolver`."""

    nG: int = 41              # target number of Fourier harmonics
    grid_nx: int = 96         # unit cell raster resolution along ax
    grid_ny: int = 96         # unit cell raster resolution along ay
    polarization: str = "average"   # "average" | "te" | "tm"
    theta_deg: float = 0.0    # incidence angle
    phi_deg: float = 0.0
    verbose: bool = False


class RCWASolver(ElectromagneticSolver):
    """Concrete :class:`ElectromagneticSolver` backed by grcwa."""

    def __init__(
        self,
        config: Optional[RCWAConfig] = None,
        dispersion: Optional[SiNDispersion] = None,
        n_superstrate: complex = 1.0 + 0.0j,
        n_substrate: complex = 1.0 + 0.0j,
    ):
        try:
            import grcwa  # noqa: F401  (import test)
        except ImportError as err:
            raise ImportError(
                "RCWASolver requires grcwa; install with `pip install 'lightsail[rcwa]'`."
            ) from err

        self.config = config or RCWAConfig()
        self.dispersion = dispersion or SiNDispersion()
        self.n_superstrate = complex(n_superstrate)
        self.n_substrate = complex(n_substrate)

        # Cheap caches to avoid recomputing ε grids and unit cells when only
        # the wavelength changes between calls on the same Structure.
        self._last_structure_id: Optional[int] = None
        self._cached_unit_cell: Optional[_UnitCell] = None
        self._cached_eps_grid: Optional[np.ndarray] = None  # shape (Nx, Ny)

    # ------------------------------------------------------------------
    # ElectromagneticSolver interface
    # ------------------------------------------------------------------

    def evaluate_reflectivity(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        r, _ = self._evaluate_rt(structure, wavelengths_nm)
        return r

    def evaluate_transmission(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        _, t = self._evaluate_rt(structure, wavelengths_nm)
        return t

    # evaluate_emissivity inherits default 1 − R − T from the base class.

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def _evaluate_rt(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        wl_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        n = wl_nm.size
        R = np.zeros(n, dtype=float)
        T = np.zeros(n, dtype=float)

        unit_cell, eps_grid = self._prepare_geometry(structure)

        # Per-wavelength loop. grcwa sim objects are single-frequency.
        for i, wl in enumerate(wl_nm):
            R[i], T[i] = self._solve_one(
                wl_nm=float(wl),
                thickness_nm=float(structure.thickness_nm),
                unit_cell=unit_cell,
                eps_grid=eps_grid,
            )
        return R, T

    def _solve_one(
        self,
        wl_nm: float,
        thickness_nm: float,
        unit_cell: "_UnitCell",
        eps_grid: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        import grcwa

        # SiN complex permittivity at this wavelength
        eps_sin = complex(self.dispersion.epsilon(wl_nm))
        eps_hole = 1.0 + 0.0j

        # Everything in µm for grcwa
        wl_um = wl_nm / 1000.0
        thick_um = thickness_nm / 1000.0
        freq = 1.0 / wl_um  # in units of 1/µm
        # grcwa requires plain Python lists for lattice vectors (not tuples).
        L1 = list(unit_cell.L1_um)
        L2 = list(unit_cell.L2_um)

        # For an unpatterned (slab) structure we can short-circuit to a
        # uniform layer — nG > ~5 becomes singular for a homogeneous film.
        uniform = eps_grid is None

        polarization_calls = _polarization_loops(self.config.polarization)
        r_sum = 0.0
        t_sum = 0.0

        for p_amp, s_amp in polarization_calls:
            sim = grcwa.obj(
                self.config.nG if not uniform else 3,
                L1,
                L2,
                freq,
                self.config.theta_deg,
                self.config.phi_deg,
                verbose=0,
            )
            sim.Add_LayerUniform(0.0, complex(self.n_superstrate) ** 2)
            if uniform:
                sim.Add_LayerUniform(thick_um, eps_sin)
            else:
                sim.Add_LayerGrid(thick_um, eps_grid.shape[0], eps_grid.shape[1])
            sim.Add_LayerUniform(0.0, complex(self.n_substrate) ** 2)

            sim.Init_Setup()

            if not uniform:
                # Replace ε values in the grid: SiN bulk + air holes
                filled = np.where(eps_grid > 0.5, eps_sin, eps_hole)
                sim.GridLayer_geteps(filled.astype(complex).flatten())

            sim.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
            try:
                Ri, Ti = sim.RT_Solve(normalize=1)
            except Exception as err:  # pragma: no cover  (numerical edge)
                logger.warning("grcwa RT_Solve failed at %.0f nm: %s", wl_nm, err)
                Ri, Ti = 0.0, 1.0

            r_sum += float(np.real(Ri))
            t_sum += float(np.real(Ti))

        n_pol = len(polarization_calls)
        R = float(np.clip(r_sum / n_pol, 0.0, 1.0))
        T = float(np.clip(t_sum / n_pol, 0.0, 1.0 - R))
        return R, T

    # ------------------------------------------------------------------
    # Geometry → RCWA grid
    # ------------------------------------------------------------------

    def _prepare_geometry(
        self,
        structure: Structure,
    ) -> Tuple["_UnitCell", Optional[np.ndarray]]:
        """Return the unit cell and ε-grid mask (``None`` if no holes)."""
        struct_id = id(structure)
        if (
            struct_id == self._last_structure_id
            and self._cached_unit_cell is not None
        ):
            return self._cached_unit_cell, self._cached_eps_grid

        unit_cell = _unit_cell_for(structure)
        if structure.has_phc and structure.holes:
            eps_grid = _rasterize_holes(
                structure=structure,
                unit_cell=unit_cell,
                nx=self.config.grid_nx,
                ny=self.config.grid_ny,
            )
        else:
            eps_grid = None  # uniform slab / metagrating placeholder

        self._last_structure_id = struct_id
        self._cached_unit_cell = unit_cell
        self._cached_eps_grid = eps_grid
        return unit_cell, eps_grid


# ---------------------------------------------------------------------------
# Unit cell definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UnitCell:
    """Minimal unit cell description in micrometers for grcwa."""

    L1_um: Tuple[float, float]
    L2_um: Tuple[float, float]
    hole_offsets_nm: Tuple[Tuple[float, float], ...]

    @property
    def area_um2(self) -> float:
        (x1, y1), (x2, y2) = self.L1_um, self.L2_um
        return abs(x1 * y2 - x2 * y1)


def _unit_cell_for(structure: Structure) -> _UnitCell:
    """Pick a primitive (or super-) unit cell for a Structure."""
    family = structure.lattice_family
    period_nm = structure.lattice_period_nm or 1000.0
    period_um = period_nm / 1000.0

    if family == LatticeFamily.TRIANGULAR:
        # 60° parallelogram: a1 = (p, 0), a2 = (p/2, p√3/2). One hole at (0, 0).
        return _UnitCell(
            L1_um=(period_um, 0.0),
            L2_um=(period_um * 0.5, period_um * np.sqrt(3.0) * 0.5),
            hole_offsets_nm=((0.0, 0.0),),
        )

    if family == LatticeFamily.HEXAGONAL:
        # Honeycomb: Bravais triangular with period p√3, two sublattice sites.
        p_bravais = period_um * np.sqrt(3.0)
        return _UnitCell(
            L1_um=(p_bravais, 0.0),
            L2_um=(p_bravais * 0.5, p_bravais * np.sqrt(3.0) * 0.5),
            hole_offsets_nm=((0.0, 0.0), (0.0, period_nm)),
        )

    if family == LatticeFamily.RECTANGULAR:
        # Orthogonal unit cell with independent x/y periods. The
        # Structure already carries period_x/period_y after the geometry
        # builds it, so use those.
        px_um = (structure.period_x_nm or period_nm) / 1000.0
        py_um = (structure.period_y_nm or period_nm) / 1000.0
        return _UnitCell(
            L1_um=(px_um, 0.0),
            L2_um=(0.0, py_um),
            hole_offsets_nm=((0.0, 0.0),),
        )

    if family == LatticeFamily.PENTAGONAL_SUPERCELL:
        # Square supercell of side `period_nm` holding five pentagon vertices.
        r = 0.32 * period_nm
        offsets = []
        for k in range(5):
            ang = 2.0 * np.pi * k / 5.0 + np.pi / 2.0
            offsets.append((r * np.cos(ang), r * np.sin(ang)))
        return _UnitCell(
            L1_um=(period_um, 0.0),
            L2_um=(0.0, period_um),
            hole_offsets_nm=tuple(offsets),
        )

    if family == LatticeFamily.DUAL_TRIANGULAR:
        # Rectangular supercell containing 2 triangular primitive cells.
        # L1 = (p, 0),  L2 = (0, p*sqrt(3)).
        # Two hole offsets: (0, 0) and (p/2, p*sqrt(3)/2).
        sqrt3 = np.sqrt(3.0)
        return _UnitCell(
            L1_um=(period_um, 0.0),
            L2_um=(0.0, period_um * sqrt3),
            hole_offsets_nm=(
                (0.0, 0.0),
                (period_nm * 0.5, period_nm * sqrt3 * 0.5),
            ),
        )

    if family == LatticeFamily.DISORDERED_TRIANGULAR:
        # 2×2 triangular supercell with jittered hole positions.
        # L1 = (2p, 0),  L2 = (p, p*sqrt(3)).
        # The 4 jittered positions are read from structure metadata where
        # ``to_structure`` has stored them under "disordered_offsets_nm".
        sqrt3 = np.sqrt(3.0)
        raw_offsets = structure.metadata.get("disordered_offsets_nm", ())
        hole_offsets_nm = tuple(
            (float(x), float(y)) for (x, y) in raw_offsets
        )
        return _UnitCell(
            L1_um=(2.0 * period_um, 0.0),
            L2_um=(period_um, period_um * sqrt3),
            hole_offsets_nm=hole_offsets_nm,
        )

    # Unknown / none: fall back to a 1 µm square dummy cell (no holes used).
    return _UnitCell(
        L1_um=(1.0, 0.0),
        L2_um=(0.0, 1.0),
        hole_offsets_nm=(),
    )


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def _rasterize_holes(
    structure: Structure,
    unit_cell: _UnitCell,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Rasterize the HoleShape(s) onto a (nx, ny) grid filling the unit cell.

    Returns a float array with 1.0 = SiN, 0.0 = hole. The solver turns this
    mask into complex permittivity per wavelength.

    If ``structure.metadata`` contains a ``"per_hole_shapes"`` key with a
    list of :class:`HoleShape` objects (one per offset in
    ``unit_cell.hole_offsets_nm``), each offset is rasterized with its own
    shape. Otherwise every offset uses ``structure.holes[0].shape``
    (backward-compatible fallback).
    """
    # Sample (u, v) fractional coordinates uniformly in [0, 1).
    u = np.linspace(0.0, 1.0, nx, endpoint=False)
    v = np.linspace(0.0, 1.0, ny, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Convert to real-space (x, y) in nm by applying the lattice vectors.
    L1x_nm = unit_cell.L1_um[0] * 1000.0
    L1y_nm = unit_cell.L1_um[1] * 1000.0
    L2x_nm = unit_cell.L2_um[0] * 1000.0
    L2y_nm = unit_cell.L2_um[1] * 1000.0
    X = U * L1x_nm + V * L2x_nm
    Y = U * L1y_nm + V * L2y_nm

    # Resolve per-hole shapes or fall back to the shared shape for all holes.
    per_hole_shapes: Optional[list[HoleShape]] = structure.metadata.get(
        "per_hole_shapes"
    )
    if per_hole_shapes is not None:
        # Build one mask function per offset (may differ in shape).
        mask_fns = [_build_hole_mask_fn(s) for s in per_hole_shapes]
    else:
        # All holes share the same shape — backward-compatible path.
        shared_fn = _build_hole_mask_fn(structure.holes[0].shape)
        mask_fns = [shared_fn] * len(unit_cell.hole_offsets_nm)

    mask = np.ones((nx, ny), dtype=float)  # 1 = SiN, 0 = hole

    # The hole centers defined by the unit cell template; we wrap shifts
    # using periodic distance. Also check a 1-cell ring of periodic images
    # so that a hole straddling the boundary is rendered correctly.
    image_shifts = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    for idx, (ox_nm, oy_nm) in enumerate(unit_cell.hole_offsets_nm):
        mask_fn = mask_fns[idx]
        for (i, j) in image_shifts:
            cx = ox_nm + i * L1x_nm + j * L2x_nm
            cy = oy_nm + i * L1y_nm + j * L2y_nm
            dx = X - cx
            dy = Y - cy
            mask[mask_fn(dx, dy)] = 0.0

    return mask


def _build_hole_mask_fn(shape: HoleShape) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a callable ``(dx, dy) -> bool mask`` for points inside the shape.

    Evaluates the same rounded-polygon radial function used by
    :meth:`HoleShape.boundary`, and returns points whose distance from the
    center is below the shape boundary at that angle.
    """
    a = shape.a_nm
    b = shape.b_nm
    n_sides = shape.n_sides
    rot = np.radians(shape.rotation_deg)
    rounding = shape.corner_rounding

    cos_rot, sin_rot = np.cos(rot), np.sin(rot)
    sector = 2.0 * np.pi / n_sides
    round_cos_term = np.cos(np.pi / n_sides)

    def inside(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        # Transform to the shape's local (un-rotated) frame.
        x_loc = cos_rot * dx + sin_rot * dy
        y_loc = -sin_rot * dx + cos_rot * dy

        # Scale so that the inscribed "unit shape" has semi-axes a, b.
        u = x_loc / a
        v = y_loc / b
        r_norm = np.hypot(u, v)

        # Angle in the (u, v) frame.
        theta = np.arctan2(v, u)
        phi_rel = np.mod(theta + sector / 2.0, sector) - sector / 2.0
        r_polygon_boundary = round_cos_term / np.cos(phi_rel)
        r_shape_boundary = (1.0 - rounding) * r_polygon_boundary + rounding * 1.0

        return r_norm <= r_shape_boundary

    return inside


# ---------------------------------------------------------------------------
# Polarization helper
# ---------------------------------------------------------------------------


def _polarization_loops(mode: str) -> list[Tuple[float, float]]:
    """Return the (p_amp, s_amp) pairs to sum over.

    grcwa's ``MakeExcitationPlanewave`` takes (p_amp, p_phase, s_amp,
    s_phase). We loop explicitly and average.
    """
    if mode == "te":
        return [(0.0, 1.0)]
    if mode == "tm":
        return [(1.0, 0.0)]
    return [(1.0, 0.0), (0.0, 1.0)]  # unpolarized average
