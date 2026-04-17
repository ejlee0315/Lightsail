"""Mock solver for pipeline testing and development.

Returns physically-plausible but synthetic spectra built from a few
toy analytical models:

1. Fabry-Perot thin-film baseline from thickness and n_SiN.
2. PhC bandgap enhancement peaking near ``lattice_period * n_SiN``.
   Strength modulated by the hole fill fraction (peaks around 0.3).
3. MetaGrating diffraction bump in the NIR (around
   ``grating_period * n_SiN``), modulated by ``4 * duty * (1-duty)``.
4. MIR phonon absorption around 10.5 µm (SiN Si-N stretch), scaled
   roughly with thickness.
5. Energy conservation: R + T + A = 1, all clipped to [0, 1].

Not physically accurate — DO NOT use for design decisions. This is
strictly a debugging / pipeline-test solver.
"""

from __future__ import annotations

import numpy as np

from lightsail.geometry.base import Structure
from lightsail.simulation.base import ElectromagneticSolver


class MockSolver(ElectromagneticSolver):
    """Wavelength-dependent pseudo-physics solver."""

    def __init__(
        self,
        n_sin: float = 2.0,
        phc_resonance_factor: float = 1.1,
        mir_phonon_center_nm: float = 10_500.0,
        mir_phonon_sigma_nm: float = 2_500.0,
    ):
        self.n_sin = n_sin
        self.phc_resonance_factor = phc_resonance_factor
        self.mir_phonon_center_nm = mir_phonon_center_nm
        self.mir_phonon_sigma_nm = mir_phonon_sigma_nm

    # ------------------------------------------------------------------
    # ElectromagneticSolver interface
    # ------------------------------------------------------------------

    def evaluate_reflectivity(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        r, _, _ = self._compute_rta(structure, wavelengths_nm)
        return r

    def evaluate_transmission(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        _, t, _ = self._compute_rta(structure, wavelengths_nm)
        return t

    def evaluate_emissivity(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        _, _, a = self._compute_rta(structure, wavelengths_nm)
        return a

    # ------------------------------------------------------------------
    # Internal model
    # ------------------------------------------------------------------

    def _compute_rta(
        self,
        structure: Structure,
        wavelengths_nm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (R, T, A) arrays with R + T + A = 1."""
        wl = np.asarray(wavelengths_nm, dtype=float)
        wl = np.clip(wl, 50.0, None)  # avoid div-by-zero if wl=0 slips in

        reflectance = self._fabry_perot(structure, wl)
        reflectance = reflectance + self._phc_bandgap_boost(structure, wl)
        reflectance = reflectance + self._metagrating_boost(structure, wl)
        reflectance = np.clip(reflectance, 0.0, 1.0)

        absorption = self._mir_absorption(structure, wl)
        absorption = np.clip(absorption, 0.0, 0.95)

        # Energy conservation: cap total at 1 by reducing transmission first,
        # then shaving from reflectance if needed.
        transmittance = 1.0 - reflectance - absorption
        transmittance = np.clip(transmittance, 0.0, 1.0)

        total = reflectance + transmittance + absorption
        over = np.clip(total - 1.0, 0.0, None)
        reflectance = np.clip(reflectance - over, 0.0, 1.0)

        absorption = np.clip(1.0 - reflectance - transmittance, 0.0, 1.0)
        return reflectance, transmittance, absorption

    # ---- individual contributions -----------------------------------

    def _fabry_perot(self, structure: Structure, wl: np.ndarray) -> np.ndarray:
        """Thin-film reflectance oscillating with 2π·n·d/λ."""
        delta = 2.0 * np.pi * self.n_sin * structure.thickness_nm / wl
        r_max = ((self.n_sin - 1.0) / (self.n_sin + 1.0)) ** 2
        return r_max * 0.5 * (1.0 + np.cos(delta))

    def _phc_bandgap_boost(
        self,
        structure: Structure,
        wl: np.ndarray,
    ) -> np.ndarray:
        if not structure.has_phc or not structure.lattice_period_nm:
            return np.zeros_like(wl)

        period = structure.lattice_period_nm
        center = period * self.n_sin * self.phc_resonance_factor
        sigma = max(center * 0.18, 1.0)
        bandgap = np.exp(-0.5 * ((wl - center) / sigma) ** 2)

        # Fill fraction ~ hole_area / cell_area. Peak effect near 0.3.
        hole_area = (
            structure.holes[0].shape.area_nm2() if structure.holes else 0.0
        )
        cell_area = structure.metadata.get(
            "unit_cell_area_nm2", period * period
        )
        fill = float(np.clip(hole_area / max(cell_area, 1.0), 0.0, 0.8))
        fill_weight = max(0.0, 1.0 - abs(fill - 0.3) / 0.3)

        return 0.85 * fill_weight * bandgap

    def _metagrating_boost(
        self,
        structure: Structure,
        wl: np.ndarray,
    ) -> np.ndarray:
        if not structure.has_metagrating:
            return np.zeros_like(wl)
        period_grat = structure.metadata.get("grating_period_nm", 1500.0)
        duty = structure.metadata.get("duty_cycle", 0.5)

        center = period_grat * self.n_sin * 0.9
        sigma = 400.0
        resp = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
        strength = 4.0 * duty * (1.0 - duty)  # max at duty=0.5
        return 0.20 * strength * resp

    def _mir_absorption(
        self,
        structure: Structure,
        wl: np.ndarray,
    ) -> np.ndarray:
        gaussian = np.exp(
            -0.5 * ((wl - self.mir_phonon_center_nm) / self.mir_phonon_sigma_nm) ** 2
        )
        base = 0.70 * gaussian
        thickness_scale = np.clip(structure.thickness_nm / 500.0, 0.3, 2.0)
        return base * thickness_scale
