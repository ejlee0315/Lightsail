"""Linearized 2×2 stiffness matrix for the outer metagrating zone.

Implements docx Eq. 4.7 by aggregating the FMM-derived ``∂C_pr,1/∂θ``
across the radial bins of a :class:`MetaGrating`. Under uniform
plane-wave illumination of an axisymmetric sail, translation-
invariance gives ``k_xx = k_θx = 0`` exactly; the meaningful
coefficients are ``k_xθ`` and ``k_θθ``, which couple a tilt to a
restoring lateral force and torque.

Physics
-------
For a periodic grating illuminated at incidence angle θ_in, the
lateral momentum flux removed from the photon stream per area is

    F_y / A_inc = -(I/c) · C_pr,1(θ_in)

so the angular force-density derivative is

    σ_xθ = ∂(F_y/A) / ∂θ_in = -(I/c) · ∂C_pr,1/∂θ_in     (Pa/rad)

Integrating σ_xθ over the metagrating ring zone (annular area
``A_ring = π(r_outer² − r_inner²)``) yields ``k_xθ`` (N/rad).
The torque comes from the lever arm of the ring around the sail
center: ``k_θθ = k_xθ · r_mean`` (N·m/rad).

A real Gaussian beam would generate non-zero ``k_xx`` and ``k_θx``
through the lateral intensity gradient; we expose the plane-wave
result here as the physically interpretable lower bound and leave
beam-profile-aware extensions to a downstream wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lightsail.geometry.metagrating import MetaGrating
from lightsail.materials import SiNDispersion
from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    aggregate_metagrating_response,
)

C_LIGHT_M_PER_S = 2.99792458e8
DEFAULT_INTENSITY_W_PER_M2 = 1.0e10  # Starshot-class 10 GW/m²


@dataclass
class StiffnessResult:
    """Container for the 2×2 stiffness matrix and its FMM provenance."""

    k_xx_N_per_m: float
    k_xtheta_N_per_rad: float
    k_thetax_Nm_per_m: float
    k_thetatheta_Nm_per_rad: float
    sigma_xtheta_Pa_per_rad: float
    ring_area_m2: float
    ring_mean_radius_m: float
    intensity_W_per_m2: float
    fmm_aggregate: dict

    @property
    def restoring(self) -> bool:
        """Whether the angular restoring torque is positive (stable tilt)."""
        return self.k_thetatheta_Nm_per_rad > 0.0

    def as_matrix(self) -> np.ndarray:
        """Return the 2×2 matrix [[k_xx, k_xθ], [k_θx, k_θθ]]."""
        return np.array(
            [
                [self.k_xx_N_per_m, self.k_xtheta_N_per_rad],
                [self.k_thetax_Nm_per_m, self.k_thetatheta_Nm_per_rad],
            ],
            dtype=float,
        )


def compute_stiffness_matrix(
    metagrating: MetaGrating,
    nir_band_nm: tuple[float, float] = (1550.0, 1850.0),
    nir_n_points: int = 5,
    n_radial_bins: int = 5,
    intensity_W_per_m2: float = DEFAULT_INTENSITY_W_PER_M2,
    dtheta_deg: float = 1.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> StiffnessResult:
    """Compute the linearized stiffness matrix for an outer metagrating.

    Parameters
    ----------
    metagrating : MetaGrating
        Outer-ring geometry whose ``inner_radius_nm`` and
        ``outer_radius_nm`` define the annular area.
    nir_band_nm, nir_n_points : NIR Doppler band sampling.
    n_radial_bins : Radial slicing for the FMM aggregation; the local
        period is modulated by ``metagrating.curvature``.
    intensity_W_per_m2 : Local laser intensity (Starshot ~ 10 GW/m²).
    dtheta_deg : Centered finite-difference step for ``∂C_pr,1/∂θ``.
    """
    period_nm = float(metagrating.grating_period_nm)
    duty = float(metagrating.duty_cycle)
    thickness_nm = float(metagrating.thickness_nm)
    curvature = float(metagrating.curvature)

    r_inner_m = float(metagrating.inner_radius_nm) * 1e-9
    r_outer_m = float(metagrating.outer_radius_nm) * 1e-9
    r_mean_m = 0.5 * (r_inner_m + r_outer_m)
    ring_area_m2 = float(np.pi * (r_outer_m ** 2 - r_inner_m ** 2))

    wls = np.linspace(nir_band_nm[0], nir_band_nm[1], nir_n_points)
    agg = aggregate_metagrating_response(
        grating_period_nm=period_nm,
        duty_cycle=duty,
        thickness_nm=thickness_nm,
        wavelengths_nm=wls,
        curvature=curvature,
        n_radial_bins=n_radial_bins,
        theta_center_deg=0.0,
        dtheta_deg=dtheta_deg,
        dispersion=dispersion,
        config=config,
    )

    sigma_xtheta = -(intensity_W_per_m2 / C_LIGHT_M_PER_S) * float(
        agg["mean_dC_pr_1_dtheta"]
    )
    k_xtheta = sigma_xtheta * ring_area_m2
    k_thetatheta = k_xtheta * r_mean_m

    return StiffnessResult(
        k_xx_N_per_m=0.0,
        k_xtheta_N_per_rad=k_xtheta,
        k_thetax_Nm_per_m=0.0,
        k_thetatheta_Nm_per_rad=k_thetatheta,
        sigma_xtheta_Pa_per_rad=sigma_xtheta,
        ring_area_m2=ring_area_m2,
        ring_mean_radius_m=r_mean_m,
        intensity_W_per_m2=float(intensity_W_per_m2),
        fmm_aggregate=agg,
    )
