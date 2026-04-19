"""SiC (silicon carbide) Reststrahlen MIR dispersion.

4H-SiC ordinary-ray Lorentz dispersion (single phonon oscillator):

    ε(ω) = ε_∞ × [1 + (ω_LO² − ω_TO²) / (ω_TO² − ω² − i γ ω)]

with::

    ε_∞   = 6.7        (high-frequency limit)
    ω_TO  = 797 cm⁻¹   (TO phonon, λ ≈ 12.55 µm)
    ω_LO  = 973 cm⁻¹   (LO phonon, λ ≈ 10.28 µm)
    γ     = 4.76 cm⁻¹  (damping)

Between ω_TO and ω_LO (10.3 → 12.6 µm) the real part of ε is negative
— the Reststrahlen band — and SiC behaves like a polar-phonon
"metal" with strong absorption and high reflection. Just outside this
band SiC is a normal high-index dielectric (n ≈ √6.7 ≈ 2.6) with
near-zero loss in NIR.

For an outside-the-Reststrahlen baseline (e.g. 1.55 µm laser):
n ≈ 2.6, k ≈ 0 → mostly transparent, modest reflection. This makes
SiC a candidate "thermal-functional layer" that doesn't strongly
perturb NIR while providing strong MIR emissivity from 10–13 µm.

References
----------
* Mutschke et al., "Infrared properties of SiC particles", Astron.
  Astrophys. 345, 187 (1999).
* Spitzer et al., "Infrared properties of hexagonal silicon carbide",
  Phys. Rev. 113, 127 (1959).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Physical constants
C_LIGHT_M_PER_S = 2.99792458e8

# 4H-SiC parameters (Mutschke 1999)
_DEFAULT_EPS_INF = 6.7
_DEFAULT_OMEGA_TO_CM_INV = 797.0
_DEFAULT_OMEGA_LO_CM_INV = 973.0
_DEFAULT_GAMMA_CM_INV = 4.76


def _cm_inv_to_rad_per_s(nu_cm_inv: float) -> float:
    """Convert wavenumber (cm⁻¹) to angular frequency (rad/s)."""
    return 2.0 * np.pi * C_LIGHT_M_PER_S * (nu_cm_inv * 100.0)


@dataclass
class SiCDispersion:
    """Single-oscillator Lorentz dispersion for the 4H-SiC Reststrahlen."""

    eps_inf: float = _DEFAULT_EPS_INF
    omega_TO_cm: float = _DEFAULT_OMEGA_TO_CM_INV
    omega_LO_cm: float = _DEFAULT_OMEGA_LO_CM_INV
    gamma_cm: float = _DEFAULT_GAMMA_CM_INV

    def epsilon(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Complex permittivity ε(ω) at one or many wavelengths."""
        wl_m = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) * 1e-9
        omega = 2.0 * np.pi * C_LIGHT_M_PER_S / wl_m
        omega_TO = _cm_inv_to_rad_per_s(self.omega_TO_cm)
        omega_LO = _cm_inv_to_rad_per_s(self.omega_LO_cm)
        gamma = _cm_inv_to_rad_per_s(self.gamma_cm)
        delta = omega_LO ** 2 - omega_TO ** 2
        denom = omega_TO ** 2 - omega ** 2 - 1j * gamma * omega
        eps = self.eps_inf * (1.0 + delta / denom)
        return eps.reshape(np.shape(wavelength_nm))

    def n(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        return np.real(np.sqrt(self.epsilon(wavelength_nm)))

    def k(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        return np.imag(np.sqrt(self.epsilon(wavelength_nm)))

    def epsilon_callable(self) -> Callable[[float], complex]:
        """Return ``λ_nm → ε`` callable for ``LayerSpec.eps_callable``."""
        return lambda wl: complex(self.epsilon(wl))


def sic_epsilon(wavelength_nm: float) -> complex:
    """Free-function convenience: ε(λ_nm) for default 4H-SiC."""
    return complex(SiCDispersion().epsilon(wavelength_nm))


# Density used for mass accounting (g/cm³ → kg/m³)
SIC_DENSITY_KG_PER_M3 = 3210.0
