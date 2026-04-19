"""Multi-layer RCWA solver — single-layer SiN PhC + extra uniform layers.

Extends :class:`RCWASolver` with arbitrary uniform layers stacked above
or below the central PhC slab. The motivation is the docx-recommended
backside thermal-functional layer (e.g. CVD graphene) that should
sit on the *space-facing* side so it absorbs only transmitted laser
power and therefore does not perturb the propulsion-band reflectance.

Usage::

    layered = LayeredRCWASolver(
        config=RCWAConfig(nG=41),
        layers_below=[
            LayerSpec(
                thickness_nm=10 * 0.34,           # 10 graphene monolayers
                eps_callable=graphene_epsilon,    # ε(λ_nm)
                name="graphene_x10",
            ),
        ],
    )
    R = layered.evaluate_reflectivity(structure, wavelengths_nm)

When both ``layers_above`` and ``layers_below`` are empty,
:class:`LayeredRCWASolver` is numerically identical to
:class:`RCWASolver` (verified in the smoke tests).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from lightsail.materials import SiNDispersion
from lightsail.simulation.rcwa_solver import (
    RCWAConfig,
    RCWASolver,
    _polarization_loops,
)

logger = logging.getLogger(__name__)


EpsCallable = Callable[[float], complex]


@dataclass
class LayerSpec:
    """One extra uniform layer in the stack.

    Provide either ``eps_callable`` (function of wavelength_nm → complex ε)
    or ``eps_constant`` (frequency-independent value). ``eps_callable``
    wins if both are set.
    """

    thickness_nm: float
    eps_callable: Optional[EpsCallable] = None
    eps_constant: Optional[complex] = None
    name: str = "layer"

    def epsilon(self, wavelength_nm: float) -> complex:
        if self.eps_callable is not None:
            return complex(self.eps_callable(wavelength_nm))
        if self.eps_constant is not None:
            return complex(self.eps_constant)
        raise ValueError(
            f"LayerSpec '{self.name}': set eps_callable or eps_constant."
        )

    @property
    def thickness_um(self) -> float:
        return float(self.thickness_nm) / 1000.0


@dataclass
class PatternedLayerSpec:
    """Patterned (2D) layer in the stack — for backside MIR absorber etc.

    The unit cell is shared with the base PhC's lattice (so the patterns
    co-exist in the same RCWA simulation). The ``eps_grid`` is a binary
    mask (1 = material, 0 = air) of shape ``(nx, ny)`` where (nx, ny) match
    the base solver's ``grid_nx`` / ``grid_ny``. ``eps_callable`` provides
    the *material* permittivity ε_mat(λ); air = 1 elsewhere.

    For multi-size patches (broadband), use a super-cell unit cell on
    the base solver and place differently-sized rectangular patches in
    the eps_grid.
    """

    thickness_nm: float
    eps_grid: np.ndarray                          # (nx, ny) binary mask
    eps_callable: Optional[EpsCallable] = None
    eps_mat_constant: Optional[complex] = None
    name: str = "patterned"

    def epsilon_mat(self, wavelength_nm: float) -> complex:
        if self.eps_callable is not None:
            return complex(self.eps_callable(wavelength_nm))
        if self.eps_mat_constant is not None:
            return complex(self.eps_mat_constant)
        raise ValueError(
            f"PatternedLayerSpec '{self.name}': set eps_callable or eps_mat_constant."
        )

    @property
    def thickness_um(self) -> float:
        return float(self.thickness_nm) / 1000.0

    @property
    def fill_fraction(self) -> float:
        return float(np.mean(self.eps_grid > 0.5))


class LayeredRCWASolver(RCWASolver):
    """RCWASolver with extra uniform layers above and/or below the PhC.

    The base PhC layer (rasterized from ``Structure``) sits in the
    middle. ``layers_above`` are inserted between the superstrate and
    the PhC; ``layers_below`` between the PhC and the substrate.
    """

    def __init__(
        self,
        config: Optional[RCWAConfig] = None,
        dispersion: Optional[SiNDispersion] = None,
        n_superstrate: complex = 1.0 + 0.0j,
        n_substrate: complex = 1.0 + 0.0j,
        layers_above: Optional[List] = None,
        layers_below: Optional[List] = None,
    ):
        """Layers can be ``LayerSpec`` (uniform) or ``PatternedLayerSpec``."""
        super().__init__(
            config=config,
            dispersion=dispersion,
            n_superstrate=n_superstrate,
            n_substrate=n_substrate,
        )
        self.layers_above = list(layers_above or [])
        self.layers_below = list(layers_below or [])

    def _solve_one(
        self,
        wl_nm: float,
        thickness_nm: float,
        unit_cell,
        eps_grid,
    ):
        import grcwa

        eps_sin = complex(self.dispersion.epsilon(wl_nm))
        eps_hole = 1.0 + 0.0j

        wl_um = wl_nm / 1000.0
        thick_um = thickness_nm / 1000.0
        freq = 1.0 / wl_um
        L1 = list(unit_cell.L1_um)
        L2 = list(unit_cell.L2_um)

        uniform = eps_grid is None
        polarization_calls = _polarization_loops(self.config.polarization)
        r_sum = 0.0
        t_sum = 0.0

        # Patterned-back-layer paths require nG > 3 even if PhC is uniform.
        any_patterned_back = any(
            isinstance(l, PatternedLayerSpec) for l in self.layers_below
        )
        any_patterned_above = any(
            isinstance(l, PatternedLayerSpec) for l in self.layers_above
        )

        for p_amp, s_amp in polarization_calls:
            nG_use = self.config.nG if (not uniform or any_patterned_back or any_patterned_above) else 3
            sim = grcwa.obj(
                nG_use,
                L1, L2, freq,
                self.config.theta_deg, self.config.phi_deg,
                verbose=0,
            )
            sim.Add_LayerUniform(0.0, complex(self.n_superstrate) ** 2)
            grid_layers_to_fill = []  # (layer_obj, grid)
            for layer in self.layers_above:
                if isinstance(layer, PatternedLayerSpec):
                    sim.Add_LayerGrid(
                        layer.thickness_um,
                        layer.eps_grid.shape[0], layer.eps_grid.shape[1],
                    )
                    grid_layers_to_fill.append((layer, "above"))
                else:
                    sim.Add_LayerUniform(layer.thickness_um, layer.epsilon(wl_nm))
            if uniform:
                sim.Add_LayerUniform(thick_um, eps_sin)
            else:
                sim.Add_LayerGrid(thick_um, eps_grid.shape[0], eps_grid.shape[1])
            for layer in self.layers_below:
                if isinstance(layer, PatternedLayerSpec):
                    sim.Add_LayerGrid(
                        layer.thickness_um,
                        layer.eps_grid.shape[0], layer.eps_grid.shape[1],
                    )
                    grid_layers_to_fill.append((layer, "below"))
                else:
                    sim.Add_LayerUniform(layer.thickness_um, layer.epsilon(wl_nm))
            sim.Add_LayerUniform(0.0, complex(self.n_substrate) ** 2)

            sim.Init_Setup()
            # grcwa requires a SINGLE concatenated epsilon array covering
            # ALL grid layers in stack order (top → bottom).
            grid_eps_concat = []
            for layer, pos in grid_layers_to_fill:
                if pos == "above":
                    eps_mat = layer.epsilon_mat(wl_nm)
                    filled = np.where(layer.eps_grid > 0.5, eps_mat, 1.0 + 0.0j)
                    grid_eps_concat.append(filled.astype(complex).flatten())
            if not uniform:
                filled = np.where(eps_grid > 0.5, eps_sin, eps_hole)
                grid_eps_concat.append(filled.astype(complex).flatten())
            for layer, pos in grid_layers_to_fill:
                if pos == "below":
                    eps_mat = layer.epsilon_mat(wl_nm)
                    filled = np.where(layer.eps_grid > 0.5, eps_mat, 1.0 + 0.0j)
                    grid_eps_concat.append(filled.astype(complex).flatten())
            if grid_eps_concat:
                sim.GridLayer_geteps(np.concatenate(grid_eps_concat))

            sim.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
            try:
                Ri, Ti = sim.RT_Solve(normalize=1)
            except Exception as err:  # pragma: no cover
                logger.warning(
                    "LayeredRCWASolver.grcwa RT_Solve failed at %.0f nm: %s",
                    wl_nm, err,
                )
                Ri, Ti = 0.0, 1.0
            r_sum += float(np.real(Ri))
            t_sum += float(np.real(Ti))

        n_pol = len(polarization_calls)
        R = float(np.clip(r_sum / n_pol, 0.0, 1.0))
        T = float(np.clip(t_sum / n_pol, 0.0, 1.0 - R))
        return R, T

    @property
    def total_extra_thickness_nm(self) -> float:
        return float(
            sum(l.thickness_nm for l in self.layers_above + self.layers_below)
        )

    def describe_stack(self) -> str:
        rows = ["superstrate"]
        for l in self.layers_above:
            rows.append(f"  ↑ {l.name} ({l.thickness_nm:.2f} nm)")
        rows.append("  ★ PhC layer (Structure.thickness_nm)")
        for l in self.layers_below:
            rows.append(f"  ↓ {l.name} ({l.thickness_nm:.2f} nm)")
        rows.append("substrate")
        return "\n".join(rows)
