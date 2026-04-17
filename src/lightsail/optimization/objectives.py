"""Optimization objectives for Stage 1 and Stage 2 of the lightsail pipeline.

Design notes
------------

Each concrete Objective subclass:

- carries a display ``name``,
- has a ``target`` in ``{"maximize", "minimize"}``,
- carries an optional ``weight`` for later scalar weighting,
- implements ``evaluate(ctx)`` returning an :class:`ObjectiveValue`.

The ``ObjectiveContext`` is a small bundle passed through the
objective stack. It owns a spectrum cache keyed by
``(band_lo, band_hi, n_points)`` so that multiple objectives acting
on the same band (e.g. NIR reflection and NIR-based stabilization)
share one solver call.

Factories ``make_stage1_objectives`` / ``make_stage2_objectives``
consume a plain dict (YAML-safe) and build the objective list with
sensible defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from lightsail.constraints.fabrication import ConstraintResult
from lightsail.geometry.base import ParametricGeometry, Structure
from lightsail.simulation.base import ElectromagneticSolver
from lightsail.simulation.results import SimulationResult


# ---------------------------------------------------------------------------
# Context and value types
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveValue:
    """Result of evaluating one objective on one design."""

    name: str
    value: float
    target: Literal["maximize", "minimize"] = "maximize"
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def directed_value(self) -> float:
        """Value in a maximize convention (minimization gets sign-flipped)."""
        return self.value if self.target == "maximize" else -self.value


@dataclass
class ObjectiveContext:
    """Read-only view over one design, plus a solver cache.

    Objectives call ``spectrum(band_nm)`` to get a cached
    :class:`SimulationResult` for the requested band.
    """

    structure: Structure
    geometry: ParametricGeometry
    solver: ElectromagneticSolver
    constraint_result: Optional[ConstraintResult] = None
    _spectrum_cache: dict = field(default_factory=dict)

    def spectrum(
        self,
        band_nm: tuple[float, float],
        n_points: int = 30,
    ) -> SimulationResult:
        key = (float(band_nm[0]), float(band_nm[1]), int(n_points))
        cache = self._spectrum_cache
        if key not in cache:
            wl = np.linspace(band_nm[0], band_nm[1], n_points)
            cache[key] = self.solver.compute_spectrum(self.structure, wl)
        return cache[key]


# ---------------------------------------------------------------------------
# Objective ABC
# ---------------------------------------------------------------------------


class Objective(ABC):
    """Abstract objective. Stores metadata and implements ``evaluate``."""

    name: str
    target: Literal["maximize", "minimize"] = "maximize"
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        ...


# ---------------------------------------------------------------------------
# Concrete objectives — optical
# ---------------------------------------------------------------------------


class NIRReflectivityObjective(Objective):
    """Mean + min-mixed NIR reflectivity (maximize).

    score = (w_mean * mean(R) + w_min * min(R)) / (w_mean + w_min)

    Using a min-weighted mixture pushes the optimizer away from
    designs with narrow reflection peaks inside the NIR band.
    """

    def __init__(
        self,
        band_nm: tuple[float, float] = (1350.0, 1650.0),
        n_points: int = 30,
        mean_weight: float = 0.7,
        min_weight: float = 0.3,
        weight: float = 1.0,
        name: str = "nir_reflectance",
    ):
        self.band_nm = (float(band_nm[0]), float(band_nm[1]))
        self.n_points = int(n_points)
        self.mean_weight = float(mean_weight)
        self.min_weight = float(min_weight)
        self.weight = float(weight)
        self.name = name
        self.target = "maximize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        result = ctx.spectrum(self.band_nm, self.n_points)
        r = result.reflectance
        mean_r = float(r.mean())
        min_r = float(r.min())
        w_sum = self.mean_weight + self.min_weight
        if w_sum <= 0:
            w_sum = 1.0
        score = (self.mean_weight * mean_r + self.min_weight * min_r) / w_sum
        return ObjectiveValue(
            name=self.name,
            value=float(np.clip(score, 0.0, 1.0)),
            target=self.target,
            weight=self.weight,
            metadata={"mean_R": mean_r, "min_R": min_r},
        )


class AccelerationTimeObjective(Objective):
    """Relativistic lightsail acceleration time T (MINIMIZE).

    Directly evaluates the FOM used by the Starshot community::

        T = (m_t c^2 / (2 I A)) * ∫_0^{β_f} γ(β)^3 / R[λ(β)]
                                          × (1+β)/(1-β) dβ

    where ``R[λ(β)]`` is the reflectance at the Doppler-shifted sail-frame
    wavelength ``λ = λ_launch × sqrt((1+β)/(1-β))``.

    This replaces the :class:`NIRReflectivityObjective` proxy with the
    actual mission metric. The optimizer learns that high R near the
    launch wavelength (low β) matters more than broadband R, because
    the γ^3 weighting concentrates most of the integrand at small β.

    The returned value is in **minutes** so BO sees an O(10) number.
    """

    def __init__(
        self,
        launch_wavelength_nm: float = 1550.0,
        beta_final: float = 0.2,
        laser_intensity_W_m2: float = 1.0e10,
        sail_area_m2: float = 10.0,
        payload_mass_kg: float = 1.0e-3,
        material_density_kg_m3: float = 3100.0,
        n_points: int = 30,
        weight: float = 1.0,
        name: str = "acceleration_time",
    ):
        self.launch_nm = float(launch_wavelength_nm)
        self.beta_f = float(beta_final)
        self.I = float(laser_intensity_W_m2)
        self.A = float(sail_area_m2)
        self.m_pay = float(payload_mass_kg)
        self.rho_mat = float(material_density_kg_m3)
        self.n_points = int(n_points)
        self.weight = float(weight)
        self.name = name
        self.target = "minimize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        from scipy.integrate import quad
        from scipy.interpolate import interp1d

        c = 299_792_458.0
        lam0 = self.launch_nm
        lam_max = lam0 * np.sqrt((1.0 + self.beta_f) / (1.0 - self.beta_f))

        # Evaluate R over Doppler-shifted wavelength range
        result = ctx.spectrum((lam0, lam_max), self.n_points)
        wl = np.linspace(lam0, lam_max, self.n_points)
        R = result.reflectance
        R_interp = interp1d(
            wl, R, kind="linear",
            fill_value=(float(R[0]), float(R[-1])),
            bounds_error=False,
        )

        # Sail mass from structure
        struct = ctx.structure
        cell_area = float(struct.metadata.get("unit_cell_area_nm2", 1.0))
        hole_area = struct.holes[0].shape.area_nm2() if struct.holes else 0.0
        f_mat = max(1.0 - hole_area / cell_area, 0.01) if cell_area > 0 else 1.0
        rho_sail = self.rho_mat * struct.thickness_nm * 1e-9 * f_mat
        m_total = rho_sail * self.A + self.m_pay

        # T integral
        def integrand(beta: float) -> float:
            if beta < 1e-12:
                return 0.0
            gamma = 1.0 / np.sqrt(1.0 - beta**2)
            lam_sail = lam0 * np.sqrt((1.0 + beta) / (1.0 - beta))
            r = max(float(R_interp(lam_sail)), 0.001)
            return gamma**3 * (1.0 + beta) / (1.0 - beta) / r

        T_integral, _ = quad(integrand, 0, self.beta_f, limit=100)
        T_seconds = m_total * c**2 / (2.0 * self.I * self.A) * T_integral
        T_minutes = T_seconds / 60.0

        mean_R = float(R.mean())
        return ObjectiveValue(
            name=self.name,
            value=float(T_minutes),
            target=self.target,
            weight=self.weight,
            metadata={
                "T_seconds": float(T_seconds),
                "T_minutes": float(T_minutes),
                "mean_R_doppler": mean_R,
                "sail_mass_g": float(rho_sail * self.A * 1e3),
                "total_mass_g": float(m_total * 1e3),
            },
        )


class MIREmissivityObjective(Objective):
    """Mean MIR emissivity (maximize). ε(λ) = 1 − R(λ) − T(λ)."""

    def __init__(
        self,
        band_nm: tuple[float, float] = (8000.0, 14000.0),
        n_points: int = 30,
        weight: float = 1.0,
        name: str = "mir_emissivity",
    ):
        self.band_nm = (float(band_nm[0]), float(band_nm[1]))
        self.n_points = int(n_points)
        self.weight = float(weight)
        self.name = name
        self.target = "maximize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        result = ctx.spectrum(self.band_nm, self.n_points)
        eps = result.absorptance  # 1 - R - T via Kirchhoff
        mean_eps = float(eps.mean())
        min_eps = float(eps.min())
        return ObjectiveValue(
            name=self.name,
            value=float(np.clip(mean_eps, 0.0, 1.0)),
            target=self.target,
            weight=self.weight,
            metadata={"mean_eps": mean_eps, "min_eps": min_eps},
        )


# ---------------------------------------------------------------------------
# Concrete objectives — mass + fabrication
# ---------------------------------------------------------------------------


def _solid_area_nm2(structure: Structure) -> float:
    """SiN-covered area used for the mass proxy.

    For a PhC: ``cell_area − hole_area`` (per unit cell).
    For a metagrating zone: sum of annular ring areas.
    """
    if structure.has_phc and structure.lattice_period_nm:
        cell_area = structure.metadata.get(
            "unit_cell_area_nm2",
            structure.lattice_period_nm ** 2,
        )
        hole_area = (
            structure.holes[0].shape.area_nm2() if structure.holes else 0.0
        )
        return float(max(cell_area - hole_area, 0.0))
    if structure.has_metagrating:
        total = 0.0
        for ring in structure.rings:
            total += np.pi * (
                ring.outer_radius_nm ** 2 - ring.inner_radius_nm ** 2
            )
        return float(total)
    return 0.0


class MassAndFabPenaltyObjective(Objective):
    """Stage 1 combined mass + fabrication penalty (minimize).

    - Mass proxy: ``thickness_nm × solid_area`` (nm³), normalized by a
      reference scale so the reported value is ~O(1).
    - Fabrication penalty: dimensionless aggregate from
      :class:`ConstraintResult` attached to the context.

    Returned value is a linear combination with configurable weights:
    ``mass_weight * mass_norm + fab_weight * fab_penalty``.
    """

    def __init__(
        self,
        mass_weight: float = 0.3,
        fab_weight: float = 0.7,
        mass_reference_nm3: float = 5.0e8,
        weight: float = 1.0,
        name: str = "mass_and_fab",
    ):
        self.mass_weight = float(mass_weight)
        self.fab_weight = float(fab_weight)
        self.mass_reference_nm3 = float(mass_reference_nm3)
        self.weight = float(weight)
        self.name = name
        self.target = "minimize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        area = _solid_area_nm2(ctx.structure)
        mass_nm3 = ctx.structure.thickness_nm * area
        mass_norm = float(mass_nm3 / max(self.mass_reference_nm3, 1e-9))

        fab = 0.0
        if ctx.constraint_result is not None:
            fab = float(ctx.constraint_result.penalty)

        score = self.mass_weight * mass_norm + self.fab_weight * fab
        return ObjectiveValue(
            name=self.name,
            value=float(score),
            target=self.target,
            weight=self.weight,
            metadata={"mass_norm": mass_norm, "fab_penalty": fab},
        )


class SailArealDensityObjective(Objective):
    """Sail areal density (g/m^2), MINIMIZE.

    Independent of unit cell size — physical areal density that enters
    the relativistic lightsail acceleration equation (paper Eq. 3):

        rho_s [kg/m^2] = rho_material * thickness * material_fraction

    where ``material_fraction = 1 - hole_area / unit_cell_area``.
    The returned objective value is in **g/m^2** so BO sees an O(1)
    number (Starshot 2016 aspirational target ≈ 0.1 g/m^2 for 10 m^2
    × 1 g sail; typical PhC pentagonal at t=200 nm × Af=0.6 ≈ 0.37 g/m^2;
    our current thick triangular best at t=688 nm × Af=0.475 ≈ 1.0 g/m^2).

    Unlike :class:`MassAndFabPenaltyObjective`, this class does **not**
    bundle a fabrication term — keep the fabrication penalty as a
    separate objective so BO can trade mass and fab against each other
    explicitly.
    """

    def __init__(
        self,
        material_density_kg_m3: float = 3100.0,  # SiN default
        weight: float = 1.0,
        name: str = "sail_areal_density",
    ):
        self.material_density = float(material_density_kg_m3)
        self.weight = float(weight)
        self.name = name
        self.target = "minimize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        struct = ctx.structure
        solid_nm2 = _solid_area_nm2(struct)
        cell_area_nm2 = float(
            struct.metadata.get(
                "unit_cell_area_nm2",
                (struct.lattice_period_nm or 1.0) ** 2,
            )
        )
        if cell_area_nm2 <= 0.0:
            material_fraction = 1.0
        else:
            material_fraction = solid_nm2 / cell_area_nm2
        thickness_m = struct.thickness_nm * 1e-9
        areal_kg_m2 = self.material_density * thickness_m * material_fraction
        areal_g_m2 = areal_kg_m2 * 1e3
        return ObjectiveValue(
            name=self.name,
            value=float(areal_g_m2),
            target=self.target,
            weight=self.weight,
            metadata={
                "material_fraction": float(material_fraction),
                "areal_kg_m2": float(areal_kg_m2),
                "thickness_nm": float(struct.thickness_nm),
            },
        )


class FabricationPenaltyObjective(Objective):
    """Pure fabrication penalty (minimize). No mass term."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "fabrication_penalty",
    ):
        self.weight = float(weight)
        self.name = name
        self.target = "minimize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        fab = 0.0
        if ctx.constraint_result is not None:
            fab = float(ctx.constraint_result.penalty)
        return ObjectiveValue(
            name=self.name,
            value=float(fab),
            target=self.target,
            weight=self.weight,
            metadata={},
        )


# ---------------------------------------------------------------------------
# Stage 2 — stabilization proxy
# ---------------------------------------------------------------------------


class StabilizationProxy(ABC):
    """Interface for stabilization-score proxies.

    Implementations return a tuple ``(score, metadata)`` where
    ``score`` is a float in [0, 1]. The objective wrapper
    :class:`StabilizationProxyObjective` handles integration into
    the objective stack. Swap proxies for FDTD/RCWA-based force
    calculations without changing the rest of the pipeline.
    """

    @abstractmethod
    def score(self, ctx: ObjectiveContext) -> tuple[float, dict]:
        ...


class AsymmetryStabilizationProxy(StabilizationProxy):
    """Asymmetry-driven proxy.

    score ≈ NIR_reflectivity × warp_score × duty_weight

    - ``warp_score`` grows with mean(|curvature| + |asymmetry|)
      across the metagrating rings (saturating near 0.4).
    - ``duty_weight`` = 4·duty·(1−duty), peaking at 0.5.
    - The overall score is gated by NIR reflectivity: a metagrating
      cannot stabilize anything it doesn't actually reflect.
    """

    def __init__(
        self,
        nir_band_nm: tuple[float, float] = (1350.0, 1650.0),
        nir_n_points: int = 15,
    ):
        self.nir_band_nm = (float(nir_band_nm[0]), float(nir_band_nm[1]))
        self.nir_n_points = int(nir_n_points)

    def score(self, ctx: ObjectiveContext) -> tuple[float, dict]:
        structure = ctx.structure
        if not structure.has_metagrating:
            return 0.0, {"reason": "no rings"}

        nir_result = ctx.spectrum(self.nir_band_nm, self.nir_n_points)
        nir_R = float(nir_result.reflectance.mean())

        warps = [abs(r.curvature) + abs(r.asymmetry) for r in structure.rings]
        mean_warp = float(np.mean(warps)) if warps else 0.0
        warp_score = float(np.clip(mean_warp / 0.4, 0.0, 1.0))

        duty = float(structure.metadata.get("duty_cycle", 0.5))
        duty_weight = 4.0 * duty * (1.0 - duty)

        raw = nir_R * warp_score * duty_weight
        return float(np.clip(raw, 0.0, 1.0)), {
            "nir_R": nir_R,
            "mean_warp": mean_warp,
            "duty_weight": duty_weight,
        }


class RadialMomentumProxy(StabilizationProxy):
    """Radial-momentum-redistribution proxy.

    Uses the overlap between grating period and NIR wavelength to
    estimate how strongly the metagrating diffracts into non-zeroth
    orders, combined with the asymmetry amplitude that biases
    diffraction lobes and the NIR reflectivity that gates the
    available optical force.
    """

    def __init__(
        self,
        nir_band_nm: tuple[float, float] = (1350.0, 1650.0),
        nir_n_points: int = 15,
        period_tolerance_nm: float = 500.0,
    ):
        self.nir_band_nm = (float(nir_band_nm[0]), float(nir_band_nm[1]))
        self.nir_n_points = int(nir_n_points)
        self.period_tolerance_nm = float(period_tolerance_nm)

    def score(self, ctx: ObjectiveContext) -> tuple[float, dict]:
        structure = ctx.structure
        if not structure.has_metagrating:
            return 0.0, {"reason": "no rings"}

        period = float(structure.metadata.get("grating_period_nm", 1500.0))
        center_wl = 0.5 * (self.nir_band_nm[0] + self.nir_band_nm[1])
        diffraction = float(
            np.exp(-((period - center_wl) / self.period_tolerance_nm) ** 2)
        )

        mean_asym = float(
            np.mean([abs(r.asymmetry) for r in structure.rings])
        )
        asym_score = float(np.clip(mean_asym / 0.2, 0.0, 1.0))

        nir_result = ctx.spectrum(self.nir_band_nm, self.nir_n_points)
        nir_R = float(nir_result.reflectance.mean())

        raw = nir_R * (0.3 + 0.7 * (0.5 * diffraction + 0.5 * asym_score))
        return float(np.clip(raw, 0.0, 1.0)), {
            "diffraction": diffraction,
            "mean_asymmetry": mean_asym,
            "nir_R": nir_R,
        }


class StabilizationProxyObjective(Objective):
    """Wraps any :class:`StabilizationProxy` as an Objective."""

    def __init__(
        self,
        proxy: StabilizationProxy | None = None,
        mode: str = "asymmetry",
        nir_band_nm: tuple[float, float] = (1350.0, 1650.0),
        weight: float = 1.0,
        name: str = "stabilization",
    ):
        if proxy is None:
            if mode == "asymmetry":
                proxy = AsymmetryStabilizationProxy(nir_band_nm=nir_band_nm)
            elif mode == "radial_momentum":
                proxy = RadialMomentumProxy(nir_band_nm=nir_band_nm)
            else:
                raise ValueError(f"Unknown stabilization mode: {mode}")
        self.proxy = proxy
        self.weight = float(weight)
        self.name = name
        self.target = "maximize"

    def evaluate(self, ctx: ObjectiveContext) -> ObjectiveValue:
        value, meta = self.proxy.score(ctx)
        return ObjectiveValue(
            name=self.name,
            value=float(value),
            target=self.target,
            weight=self.weight,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Stage factories — config-driven
# ---------------------------------------------------------------------------


def _band(cfg: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = cfg.get(key, default)
    return float(raw[0]), float(raw[1])


def make_stage1_objectives(config: dict | None = None) -> list[Objective]:
    """Build the Stage 1 objective list from a plain config dict.

    The Stage 1 objective set focuses on the two primary physical goals
    plus a pure fabrication penalty as a soft constraint:

    - ``nir_reflectance`` ↑ (mean + min mixture on 1350–1650 nm)
    - ``mir_emissivity``  ↑ (mean ε on 8000–14000 nm)
    - ``fabrication_penalty`` ↓ (ConstraintResult.penalty, no mass term)

    Mass is *not* an optimization objective anymore — thin membranes are
    preferred by the physics (SiN is nearly lossless in NIR) and heavy
    membranes are penalized through fab constraints if they violate
    feature/gap limits. If a mass objective is needed again later,
    ``MassAndFabPenaltyObjective`` is still available in this module.

    Expected top-level keys (all optional):

    - ``nir_reflectance``: ``{band_nm, n_points, mean_weight, min_weight, weight}``
    - ``mir_emissivity``:  ``{band_nm, n_points, weight}``
    - ``fabrication_penalty``: ``{weight}``
    """
    cfg = config or {}

    nir_cfg = cfg.get("nir_reflectance", {}) or {}
    mir_cfg = cfg.get("mir_emissivity", {}) or {}
    fab_cfg = cfg.get("fabrication_penalty", {}) or {}

    objectives: list[Objective] = [
        NIRReflectivityObjective(
            band_nm=_band(nir_cfg, "band_nm", (1350.0, 1650.0)),
            n_points=int(nir_cfg.get("n_points", 30)),
            mean_weight=float(nir_cfg.get("mean_weight", 0.7)),
            min_weight=float(nir_cfg.get("min_weight", 0.3)),
            weight=float(nir_cfg.get("weight", 1.0)),
        ),
        MIREmissivityObjective(
            band_nm=_band(mir_cfg, "band_nm", (8000.0, 14000.0)),
            n_points=int(mir_cfg.get("n_points", 30)),
            weight=float(mir_cfg.get("weight", 1.0)),
        ),
        FabricationPenaltyObjective(
            weight=float(fab_cfg.get("weight", 0.3)),
        ),
    ]

    # Optional 4th objective — sail areal density (g/m^2, minimize).
    if "sail_areal_density" in cfg:
        sad_cfg = cfg.get("sail_areal_density") or {}
        objectives.append(
            SailArealDensityObjective(
                material_density_kg_m3=float(
                    sad_cfg.get("material_density_kg_m3", 3100.0)
                ),
                weight=float(sad_cfg.get("weight", 1.0)),
            )
        )

    # Optional — acceleration time T (minutes, minimize).
    # When present, this REPLACES nir_reflectance as the primary
    # reflectance-related objective with the actual mission FOM.
    if "acceleration_time" in cfg:
        at_cfg = cfg.get("acceleration_time") or {}
        # Replace NIR reflectivity with T-direct
        objectives = [o for o in objectives if o.name != "nir_reflectance"]
        objectives.insert(
            0,
            AccelerationTimeObjective(
                launch_wavelength_nm=float(at_cfg.get("launch_wavelength_nm", 1550.0)),
                beta_final=float(at_cfg.get("beta_final", 0.2)),
                laser_intensity_W_m2=float(at_cfg.get("laser_intensity_W_m2", 1.0e10)),
                sail_area_m2=float(at_cfg.get("sail_area_m2", 10.0)),
                payload_mass_kg=float(at_cfg.get("payload_mass_kg", 1.0e-3)),
                material_density_kg_m3=float(at_cfg.get("material_density_kg_m3", 3100.0)),
                n_points=int(at_cfg.get("n_points", 30)),
                weight=float(at_cfg.get("weight", 1.0)),
            ),
        )

    return objectives


def make_stage2_objectives(config: dict | None = None) -> list[Objective]:
    """Build the Stage 2 objective list from a plain config dict.

    Expected top-level keys (all optional):

    - ``nir_reflectance``: maintain term (lower weight than Stage 1)
    - ``mir_emissivity``: maintain term
    - ``stabilization``: ``{mode, nir_band_nm, weight}`` where
      ``mode`` is ``asymmetry`` or ``radial_momentum``
    - ``fabrication_penalty``: ``{weight}``
    """
    cfg = config or {}

    nir_cfg = cfg.get("nir_reflectance", {}) or {}
    mir_cfg = cfg.get("mir_emissivity", {}) or {}
    stab_cfg = cfg.get("stabilization", {}) or {}
    fab_cfg = cfg.get("fabrication_penalty", {}) or {}

    return [
        NIRReflectivityObjective(
            band_nm=_band(nir_cfg, "band_nm", (1350.0, 1650.0)),
            n_points=int(nir_cfg.get("n_points", 30)),
            mean_weight=float(nir_cfg.get("mean_weight", 1.0)),
            min_weight=float(nir_cfg.get("min_weight", 0.0)),
            weight=float(nir_cfg.get("weight", 0.4)),
        ),
        MIREmissivityObjective(
            band_nm=_band(mir_cfg, "band_nm", (8000.0, 14000.0)),
            n_points=int(mir_cfg.get("n_points", 30)),
            weight=float(mir_cfg.get("weight", 0.3)),
        ),
        StabilizationProxyObjective(
            mode=str(stab_cfg.get("mode", "asymmetry")),
            nir_band_nm=_band(stab_cfg, "nir_band_nm", (1350.0, 1650.0)),
            weight=float(stab_cfg.get("weight", 1.0)),
        ),
        FabricationPenaltyObjective(
            weight=float(fab_cfg.get("weight", 0.5)),
        ),
    ]
