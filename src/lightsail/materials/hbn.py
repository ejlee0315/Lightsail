"""h-BN (hexagonal boron nitride) anisotropic phonon-polariton dispersion.

h-BN is a uniaxial polar dielectric with two distinct Reststrahlen
bands:

    in-plane (E ⟂ c):    ω_TO = 1370 cm⁻¹, ω_LO = 1610 cm⁻¹  (6.2–7.3 µm)
    out-of-plane (E ‖ c): ω_TO = 760 cm⁻¹,  ω_LO = 825 cm⁻¹   (12.1–13.2 µm)

For a thin slab on a flat sail under near-normal incidence (E mostly
in-plane), the in-plane component dominates the optical response.
For oblique angles or for accurate emissivity over 8–14 µm we use an
isotropic-average single-component model with both oscillators::

    ε(ω) = ε_∞ + Σ_j (ω_LO,j² − ω_TO,j²) ε_∞ / (ω_TO,j² − ω² − i γ_j ω)

with ε_∞ ≈ 4.95 (in-plane) ≈ 4.10 (out-of-plane) and γ ≈ 5–7 cm⁻¹.

The two bands together provide *multi-resonance* MIR absorption that
covers more of the 8–14 µm thermal window than single-band materials
(SiC). Combined with h-BN's lower mass density (≈ 2.1 g/cm³ vs SiC
3.21), this is a more attractive backside thermal layer for
low-mass lightsails.

References
----------
* Caldwell et al., "Sub-diffractional volume-confined polaritons in
  the natural hyperbolic material hexagonal boron nitride",
  Nat. Commun. 5, 5221 (2014).
* Geick et al., "Normal-mode analysis of hexagonal boron nitride",
  Phys. Rev. 146, 543 (1966).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

C_LIGHT_M_PER_S = 2.99792458e8

# Geick 1966 / Caldwell 2014 averaged parameters
# (Single-component isotropic-average for simulation simplicity.)
_DEFAULT_EPS_INF = 4.5
_DEFAULT_OSCILLATORS = (
    # (ω_TO_cm⁻¹, ω_LO_cm⁻¹, γ_cm⁻¹)  — out-of-plane band
    (760.0, 825.0, 5.0),
    # in-plane band
    (1370.0, 1610.0, 7.0),
)

HBN_DENSITY_KG_PER_M3 = 2100.0


def _cm_inv_to_rad_per_s(nu_cm_inv: float) -> float:
    return 2.0 * np.pi * C_LIGHT_M_PER_S * (nu_cm_inv * 100.0)


@dataclass
class HBNDispersion:
    """Multi-oscillator Lorentz dispersion for hexagonal boron nitride."""

    eps_inf: float = _DEFAULT_EPS_INF
    oscillators: tuple = field(default_factory=lambda: _DEFAULT_OSCILLATORS)

    def epsilon(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        wl_m = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) * 1e-9
        omega = 2.0 * np.pi * C_LIGHT_M_PER_S / wl_m
        eps = np.full_like(omega, fill_value=self.eps_inf, dtype=complex)
        for ω_TO_cm, ω_LO_cm, γ_cm in self.oscillators:
            ω_TO = _cm_inv_to_rad_per_s(ω_TO_cm)
            ω_LO = _cm_inv_to_rad_per_s(ω_LO_cm)
            γ = _cm_inv_to_rad_per_s(γ_cm)
            denom = ω_TO ** 2 - omega ** 2 - 1j * γ * omega
            eps = eps + self.eps_inf * (ω_LO ** 2 - ω_TO ** 2) / denom
        return eps.reshape(np.shape(wavelength_nm))

    def n(self, wavelength_nm) -> np.ndarray:
        return np.real(np.sqrt(self.epsilon(wavelength_nm)))

    def k(self, wavelength_nm) -> np.ndarray:
        return np.imag(np.sqrt(self.epsilon(wavelength_nm)))

    def epsilon_callable(self) -> Callable[[float], complex]:
        return lambda wl: complex(self.epsilon(wl))


def hbn_epsilon(wavelength_nm: float) -> complex:
    return complex(HBNDispersion().epsilon(wavelength_nm))
