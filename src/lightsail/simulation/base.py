"""Abstract high-level electromagnetic solver interface.

The solver is the boundary between geometry and the optimization
loop. It takes a :class:`Structure` and a wavelength array and
returns wavelength-dependent optical properties.

Three primary methods are exposed:

- ``evaluate_reflectivity(structure, wavelengths_nm) -> R(λ)``
- ``evaluate_transmission(structure, wavelengths_nm) -> T(λ)``
- ``evaluate_emissivity(structure, wavelengths_nm)  -> ε(λ)``

By Kirchhoff's law at thermal equilibrium, emissivity equals
absorptance, so the default ``evaluate_emissivity`` returns
``1 - R - T``. A subclass may override it if it computes
absorption directly (e.g. from a permittivity imaginary part
integration in RCWA).

A convenience method ``compute_spectrum`` bundles R and T into a
single :class:`SimulationResult`. Concrete backends (S4, grcwa,
FDTD, ...) should override at least the two abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lightsail.geometry.base import Structure
from lightsail.simulation.results import SimulationResult


class ElectromagneticSolver(ABC):
    """Abstract EM solver over a SiN membrane Structure."""

    @abstractmethod
    def evaluate_reflectivity(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        """Return R(λ) in [0, 1] for each input wavelength."""
        ...

    @abstractmethod
    def evaluate_transmission(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        """Return T(λ) in [0, 1] for each input wavelength."""
        ...

    def evaluate_emissivity(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        """Return ε(λ) = 1 − R(λ) − T(λ), clipped to [0, 1]."""
        r = self.evaluate_reflectivity(structure, wavelengths_nm)
        t = self.evaluate_transmission(structure, wavelengths_nm)
        return np.clip(1.0 - r - t, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def compute_spectrum(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> SimulationResult:
        """Compute (R, T) together and bundle them into a SimulationResult."""
        wl = np.asarray(wavelengths_nm, dtype=float)
        r = np.asarray(self.evaluate_reflectivity(structure, wl))
        t = np.asarray(self.evaluate_transmission(structure, wl))
        return SimulationResult(
            wavelengths_nm=wl,
            reflectance=r,
            transmittance=t,
            metadata={"solver": self.__class__.__name__},
        )

    def band_mean_reflectivity(
        self,
        structure: Structure,
        band_nm: tuple[float, float],
        n_points: int = 30,
    ) -> float:
        wl = np.linspace(band_nm[0], band_nm[1], n_points)
        return float(self.evaluate_reflectivity(structure, wl).mean())

    def band_mean_emissivity(
        self,
        structure: Structure,
        band_nm: tuple[float, float],
        n_points: int = 30,
    ) -> float:
        wl = np.linspace(band_nm[0], band_nm[1], n_points)
        return float(self.evaluate_emissivity(structure, wl).mean())
