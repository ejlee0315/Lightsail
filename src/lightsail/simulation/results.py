"""Simulation result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Output from an RCWA (or mock) simulation.

    All spectra are arrays of the same length as wavelengths_nm.
    Values are in [0, 1].
    """

    wavelengths_nm: np.ndarray
    reflectance: np.ndarray
    transmittance: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def absorptance(self) -> np.ndarray:
        """Absorptance = 1 - R - T (energy conservation)."""
        return 1.0 - self.reflectance - self.transmittance

    def band_average(
        self,
        spectrum: np.ndarray,
        band_nm: tuple[float, float],
    ) -> float:
        """Average a spectrum over a wavelength band."""
        mask = (self.wavelengths_nm >= band_nm[0]) & (self.wavelengths_nm <= band_nm[1])
        if not np.any(mask):
            return 0.0
        return float(np.mean(spectrum[mask]))

    def nir_reflectance(self, band_nm: tuple[float, float] = (1350, 1650)) -> float:
        """Average reflectance in the NIR target band."""
        return self.band_average(self.reflectance, band_nm)

    def mir_emissivity(self, band_nm: tuple[float, float] = (8000, 14000)) -> float:
        """Average emissivity (≈ absorptance by Kirchhoff's law) in MIR band."""
        return self.band_average(self.absorptance, band_nm)
