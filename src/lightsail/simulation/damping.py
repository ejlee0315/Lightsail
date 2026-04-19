"""Relativistic damping force on a metagrating sail (docx Eq. 4.8).

Implements the transverse-momentum balance in the sail rest frame:

    dp_y/dt' = (D² · I / c) · [
        C_pr,2(0, λ′)
        − γ · C_pr,1(0, λ′) · (v_y / c)
        − (1/D − 1) · (∂C_pr,2/∂θ′)(0, λ′) · (v_y / v)
    ]

The first term is the static lateral momentum flux of the diffracted
beam at v_y = 0; the second is the relativistic-aberration drag; the
third is the metasurface-enhanced damping that scales with the
angular sensitivity of C_pr,2 — *the term we want to engineer*.

Inputs
------
* ``metagrating``: MetaGrating geometry (period, duty, curvature, thickness).
* ``beta``: longitudinal v / c (sets the Doppler factor and γ).
* ``v_y_per_c``: transverse velocity / c (small).
* ``lab_wavelength_nm``: lab-frame laser wavelength.
* ``intensity_W_per_m2``: local laser intensity (Starshot ~ 10 GW/m²).

Outputs
-------
Dict with the per-area momentum-rate (Pa), term breakdown (static /
aberration / metasurface drag), and the linear damping coefficient
``α_damp = -∂(p_dot_y)/∂(v_y)``. A positive α_damp means the
metagrating *damps* transverse motion (good for beam-riding).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from lightsail.geometry.metagrating import MetaGrating
from lightsail.materials import SiNDispersion
from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    aggregate_metagrating_response,
)

C_LIGHT_M_PER_S = 2.99792458e8
DEFAULT_INTENSITY_W_PER_M2 = 1.0e10


def doppler_factor(beta: float) -> float:
    """Relativistic Doppler factor for normal-incidence light on a receding sail.

    D = √((1 + β)/(1 − β))    (sail-frame λ = lab λ × D)
    """
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def lorentz_gamma(beta: float) -> float:
    return float(1.0 / np.sqrt(1.0 - beta ** 2))


def sail_frame_wavelength_nm(lab_wavelength_nm: float, beta: float) -> float:
    return float(lab_wavelength_nm) * doppler_factor(beta)


def compute_damping_force(
    metagrating: MetaGrating,
    beta: float,
    v_y_per_c: float = 1.0e-4,
    lab_wavelength_nm: float = 1550.0,
    intensity_W_per_m2: float = DEFAULT_INTENSITY_W_PER_M2,
    n_radial_bins: int = 3,
    dtheta_deg: float = 1.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> dict:
    """Evaluate docx Eq. 4.8 for one (β, v_y) operating point.

    Returns
    -------
    dict with keys::

        p_dot_y_Pa             total transverse momentum-rate per area (Pa)
        term_static_Pa         (D² I / c) · C_pr,2
        term_aberration_Pa     -(D² I / c) · γ · C_pr,1 · (v_y/c)
        term_metasurface_Pa    -(D² I / c) · (1/D − 1) · ∂C_pr,2/∂θ · (v_y/v)
        alpha_damp_Pa_per_mps  damping coefficient (positive = restoring)
        ...                    diagnostics (D, γ, λ′, C_pr,k, ∂C_pr,2/∂θ)
    """
    if abs(beta) >= 1.0:
        raise ValueError(f"beta must be in (-1, 1); got {beta}")

    D = doppler_factor(beta)
    gamma = lorentz_gamma(beta)
    lambda_prime = sail_frame_wavelength_nm(lab_wavelength_nm, beta)

    period_nm = float(metagrating.grating_period_nm)
    duty = float(metagrating.duty_cycle)
    thickness_nm = float(metagrating.thickness_nm)
    curvature = float(metagrating.curvature)

    agg = aggregate_metagrating_response(
        grating_period_nm=period_nm,
        duty_cycle=duty,
        thickness_nm=thickness_nm,
        wavelengths_nm=np.array([lambda_prime]),
        curvature=curvature,
        n_radial_bins=n_radial_bins,
        theta_center_deg=0.0,
        dtheta_deg=dtheta_deg,
        dispersion=dispersion,
        config=config,
    )
    C_pr_1 = float(agg["mean_C_pr_1"])
    C_pr_2 = float(agg["mean_C_pr_2"])
    dC_pr_2_dtheta = float(agg["mean_dC_pr_2_dtheta"])

    prefactor = (D ** 2) * intensity_W_per_m2 / C_LIGHT_M_PER_S

    term_static = prefactor * C_pr_2
    term_aber = -prefactor * gamma * C_pr_1 * v_y_per_c

    if abs(beta) < 1e-12:
        # No longitudinal motion → metasurface drag term is 0/0; take 0
        # (the (1/D − 1) factor itself vanishes at β=0, so the term is
        # well-defined as zero, but we avoid the v=0 divide here).
        term_meta = 0.0
    else:
        v_long = beta * C_LIGHT_M_PER_S
        v_y = v_y_per_c * C_LIGHT_M_PER_S
        term_meta = -prefactor * (1.0 / D - 1.0) * dC_pr_2_dtheta * (v_y / v_long)

    p_dot_y = term_static + term_aber + term_meta

    # α_damp = -∂(p_dot_y)/∂(v_y)
    # ∂term_aber/∂v_y = -prefactor · γ · C_pr_1 / c
    # ∂term_meta/∂v_y = -prefactor · (1/D − 1) · ∂C_pr,2/∂θ / v_long  (β > 0)
    alpha_aber = prefactor * gamma * C_pr_1 / C_LIGHT_M_PER_S
    if abs(beta) < 1e-12:
        alpha_meta = 0.0
    else:
        alpha_meta = prefactor * (1.0 / D - 1.0) * dC_pr_2_dtheta / (beta * C_LIGHT_M_PER_S)
    alpha_damp = alpha_aber + alpha_meta

    return {
        "p_dot_y_Pa": p_dot_y,
        "term_static_Pa": term_static,
        "term_aberration_Pa": term_aber,
        "term_metasurface_Pa": term_meta,
        "alpha_damp_Pa_per_mps": alpha_damp,
        "alpha_aberration_Pa_per_mps": alpha_aber,
        "alpha_metasurface_Pa_per_mps": alpha_meta,
        "doppler_factor_D": D,
        "lorentz_gamma": gamma,
        "lambda_prime_nm": lambda_prime,
        "C_pr_1": C_pr_1,
        "C_pr_2": C_pr_2,
        "dC_pr_2_dtheta": dC_pr_2_dtheta,
        "fmm_aggregate": agg,
    }
