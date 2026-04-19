"""Graphene surface conductivity (Drude + interband) and effective ε(λ).

Falkovsky / Hanson 2008 treatment::

    σ(ω) = σ_intra(ω) + σ_inter(ω)

with the doped/room-temperature limit ``E_F ≫ k_B T`` simplifying

    σ_intra(ω) ≈  -i · e² · E_F / (π ℏ² · (ω + i/τ))            (S)
    σ_inter(ω) ≈  e²/(4ℏ) · [Θ(ℏω − 2|E_F|)
                  + (i/π) · ln |(2|E_F| − ℏω) / (2|E_F| + ℏω)|]  (S)

For a stack of N graphene monolayers (each ``d = 0.34 nm``) treated
as a single bulk slab of thickness ``N·d``, the volume permittivity is

    ε(ω) = 1 + i · σ(ω) / (ε_0 · ω · d)

— independent of N because σ_2D scales linearly with N and so does d.
RCWA can then either model the stack as one slab of thickness ``N·d``
or as ``N`` slabs of thickness ``d`` each; both give equivalent results
at normal incidence.

References
----------
* Falkovsky, "Optical properties of graphene", J. Phys. Conf. Ser. 129, 012004 (2008)
* Hanson, "Dyadic Green's functions and guided surface waves...", J. Appl. Phys. 103, 064302 (2008)

The two checks every implementation has to clear:
* Universal absorption ``α ≈ π·e²/(ℏ·c) ≈ 0.023`` per monolayer at NIR
  (well above 2|E_F| at telecom λ for typical ``E_F = 0.3 eV``).
* MIR Drude tail dominates with ``Im(σ_intra) ∝ E_F / ω``, giving large
  imaginary ε and strong absorption for ``λ ≳ 5 µm``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Physical constants (SI)
HBAR_J_S = 1.0545718e-34
HBAR_eV_S = 6.582119569e-16
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
KB_J_PER_K = 1.380649e-23
KB_eV_PER_K = 8.617333262e-5
C_LIGHT_M_PER_S = 2.99792458e8

GRAPHENE_LAYER_THICKNESS_M = 0.34e-9  # ~0.34 nm per monolayer


@dataclass
class GrapheneConductivity:
    """Surface conductivity of graphene with intraband + interband terms.

    Parameters
    ----------
    E_F_eV : Fermi level (eV). Typical CVD graphene: 0.2–0.4 eV.
    tau_s : Intraband scattering time (s). 100 fs ≈ μ ~ 1000 cm²/Vs.
    T_K : Temperature (K).
    """

    E_F_eV: float = 0.3
    tau_s: float = 1.0e-13
    T_K: float = 300.0

    # ------------------------------------------------------------------
    def sigma_intra(self, wavelength_nm: float) -> complex:
        omega = self._omega(wavelength_nm)
        # Intraband (E_F >> kT, doped graphene):
        # σ_intra(ω) = -i · e² · E_F / (π ℏ² (ω + i/τ))
        E_F_J = self.E_F_eV * E_CHARGE
        denom = omega + 1j / self.tau_s
        return -1j * (E_CHARGE ** 2) * E_F_J / (np.pi * HBAR_J_S ** 2 * denom)

    def sigma_inter(self, wavelength_nm: float) -> complex:
        omega = self._omega(wavelength_nm)
        hbar_omega_eV = HBAR_eV_S * omega
        E_F = abs(self.E_F_eV)
        # Falkovsky simplified form with smooth log (no scattering smearing):
        # σ_inter(ω) ≈ (e² / 4ℏ) · [Θ(ℏω − 2 E_F)
        #                            + (i/π) · ln|(2 E_F − ℏω) / (2 E_F + ℏω)|]
        x = 2.0 * E_F - hbar_omega_eV
        y = 2.0 * E_F + hbar_omega_eV
        if abs(y) < 1e-30:
            log_term = 0.0
        else:
            log_term = np.log(abs(x) / abs(y))
        step = 1.0 if hbar_omega_eV > 2.0 * E_F else 0.0
        prefactor = (E_CHARGE ** 2) / (4.0 * HBAR_J_S)
        return prefactor * (step + 1j * log_term / np.pi)

    def sigma_total(self, wavelength_nm: float) -> complex:
        return self.sigma_intra(wavelength_nm) + self.sigma_inter(wavelength_nm)

    # ------------------------------------------------------------------
    def epsilon(
        self,
        wavelength_nm: float,
        layer_thickness_m: float = GRAPHENE_LAYER_THICKNESS_M,
    ) -> complex:
        """Volume permittivity for a graphene slab of thickness ``layer_thickness_m``.

        ε(ω) = 1 + i · σ(ω) / (ε_0 · ω · d_layer).
        Independent of stacked-layer count when treating a stack as one bulk slab.
        """
        omega = self._omega(wavelength_nm)
        sigma = self.sigma_total(wavelength_nm)
        return 1.0 + 1j * sigma / (EPS0 * omega * layer_thickness_m)

    def epsilon_callable(
        self,
        layer_thickness_m: float = GRAPHENE_LAYER_THICKNESS_M,
    ) -> Callable[[float], complex]:
        """Return ``λ_nm → ε`` callable suitable for ``LayerSpec.eps_callable``."""
        return lambda wl: complex(self.epsilon(wl, layer_thickness_m))

    # ------------------------------------------------------------------
    @staticmethod
    def _omega(wavelength_nm: float) -> float:
        wl_m = float(wavelength_nm) * 1e-9
        return 2.0 * np.pi * C_LIGHT_M_PER_S / wl_m

    @staticmethod
    def universal_absorption() -> float:
        """Closed-form per-monolayer absorption α = π e² / (ℏ c) ≈ 2.293%."""
        return float(np.pi * E_CHARGE ** 2 / (HBAR_J_S * C_LIGHT_M_PER_S * 4 * np.pi * EPS0))


def graphene_layer_eps(
    wavelength_nm: float,
    E_F_eV: float = 0.3,
    tau_s: float = 1.0e-13,
    T_K: float = 300.0,
    layer_thickness_m: float = GRAPHENE_LAYER_THICKNESS_M,
) -> complex:
    """Free-function convenience: ε(λ_nm) for a graphene slab."""
    return GrapheneConductivity(
        E_F_eV=E_F_eV, tau_s=tau_s, T_K=T_K
    ).epsilon(wavelength_nm, layer_thickness_m=layer_thickness_m)
