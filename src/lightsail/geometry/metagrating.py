"""Stage 2 geometry: concentric curved metagrating stabilization zone.

Continuous design parameters (optimization vector):
    grating_period_nm    # radial period = ring_width + gap
    duty_cycle           # fraction of period occupied by a ring (in [0,1])
    curvature            # 1st-order radial warping (unitless)
    asymmetry            # 2nd-order angular asymmetry (unitless)
    ring_width_um        # total radial width of the metagrating zone (µm)

Constructor-level (fixed):
    inner_radius_nm      # equals the PhC outer radius (coupling point)
    thickness_nm         # typically inherited from the PhC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lightsail.geometry.base import (
    Material,
    ParametricGeometry,
    Ring,
    Structure,
)


@dataclass
class MetaGrating(ParametricGeometry):
    """Concentric curved metagrating ring zone.

    The ring-count is a derived quantity: it's the largest integer
    such that ``n_rings * grating_period_nm <= ring_width_um * 1000``.
    Each ring has a width of ``duty_cycle * grating_period_nm`` and is
    followed by a gap of ``(1 - duty_cycle) * grating_period_nm``.

    ``curvature`` and ``asymmetry`` are stored on each Ring and
    applied at boundary-generation time (see Ring.boundary).
    """

    # --- constructor-level (fixed during optimization) --------------------
    inner_radius_nm: float = 5000.0
    thickness_nm: float = 500.0

    # --- continuous optimization variables --------------------------------
    grating_period_nm: float = 1200.0
    duty_cycle: float = 0.5
    curvature: float = 0.0
    asymmetry: float = 0.0
    ring_width_um: float = 10.0

    # ------------------------------------------------------------------
    # ParametricGeometry interface
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        return [
            "grating_period_nm",
            "duty_cycle",
            "curvature",
            "asymmetry",
            "ring_width_um",
        ]

    def param_bounds(self) -> list[tuple[float, float]]:
        return [
            (1000.0, 3000.0),  # grating_period_nm (must fit min feature + min gap)
            (0.2, 0.8),        # duty_cycle (both width and gap stay non-trivial)
            (-0.2, 0.2),       # curvature (radial warping)
            (-0.2, 0.2),       # asymmetry (angular 2nd-order)
            (2.0, 50.0),       # ring_width_um (total radial extent)
        ]

    def to_param_vector(self) -> np.ndarray:
        return np.array(
            [
                self.grating_period_nm,
                self.duty_cycle,
                self.curvature,
                self.asymmetry,
                self.ring_width_um,
            ],
            dtype=float,
        )

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        if v.size != 5:
            raise ValueError(f"MetaGrating expects 5 params, got {v.size}")
        self.grating_period_nm = float(v[0])
        self.duty_cycle = float(np.clip(v[1], 0.0, 1.0))
        self.curvature = float(v[2])
        self.asymmetry = float(v[3])
        self.ring_width_um = float(v[4])

    # ------------------------------------------------------------------
    # Ring generation
    # ------------------------------------------------------------------

    @property
    def _zone_width_nm(self) -> float:
        return self.ring_width_um * 1000.0

    @property
    def n_rings(self) -> int:
        """Derived ring count from zone width and period."""
        if self.grating_period_nm <= 0:
            return 0
        return max(1, int(np.floor(self._zone_width_nm / self.grating_period_nm)))

    def generate_rings(self) -> list[Ring]:
        """Return a list of Rings covering the metagrating zone."""
        rings: list[Ring] = []
        width = self.duty_cycle * self.grating_period_nm
        period = self.grating_period_nm
        current_inner = self.inner_radius_nm

        for _ in range(self.n_rings):
            rings.append(
                Ring(
                    inner_radius_nm=current_inner,
                    outer_radius_nm=current_inner + width,
                    curvature=self.curvature,
                    asymmetry=self.asymmetry,
                )
            )
            current_inner += period  # advance by one full period
        return rings

    def to_structure(self) -> Structure:
        rings = self.generate_rings()
        outer = rings[-1].outer_radius_nm if rings else self.inner_radius_nm
        return Structure(
            material=Material.SIN,
            thickness_nm=self.thickness_nm,
            rings=rings,
            extent_nm=2.0 * outer,
            metadata={
                "n_rings": len(rings),
                "duty_cycle": self.duty_cycle,
                "grating_period_nm": self.grating_period_nm,
            },
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def ring_width_nm(self) -> float:
        return self.duty_cycle * self.grating_period_nm

    @property
    def gap_width_nm(self) -> float:
        return (1.0 - self.duty_cycle) * self.grating_period_nm

    @property
    def outer_radius_nm(self) -> float:
        rings = self.generate_rings()
        return rings[-1].outer_radius_nm if rings else self.inner_radius_nm
