"""SiN (silicon nitride) refractive index dispersion.

Two complementary public datasets are combined:

1. **NIR / VIS** — Luke et al., "Broadband mid-infrared frustrated total
   internal reflection measurement of SiN waveguide cladding loss", Opt. Lett.
   40, 4823 (2015). A 3-term Sellmeier fit valid from 0.31 to 5.504 µm for
   stoichiometric Si3N4 thin films. Absorption is negligible in this range
   (k ~ 0). We use this for λ < 1.54 µm.

2. **MIR** — Kischkat et al., "Mid-infrared optical properties of thin films
   of aluminum oxide, titanium dioxide, silicon dioxide, aluminum nitride,
   and silicon nitride", Appl. Opt. 51, 6789 (2012). Tabulated n, k for
   stoichiometric Si3N4 over 1.54–14.29 µm (digitized for 1.54–15 µm here).
   The strong Si–N stretch absorption peaks around 10.5 µm. We use this for
   λ ≥ 1.54 µm.

Both datasets describe stoichiometric Si3N4. Real PECVD or LPCVD films can
have different Si:N ratios and hydrogen content; for precise design one
should use lab-measured ellipsometry + FTIR data. The public data here is a
reasonable starting point for optimization pipelines and is what this project
currently ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DATA_DIR = Path(__file__).resolve().parent / "data"
_KISCHKAT_CSV = _DATA_DIR / "sin_kischkat_mir.csv"

# Luke 2015 Sellmeier coefficients for Si3N4 (λ in µm):
#     n²(λ) = 1 + Σ Bi λ² / (λ² − Ci²)
# (Refractive index only — k ≈ 0 in this range.)
_LUKE_SELLMEIER = {
    "B": (3.0249, 40314.0),
    "C": (0.1353406, 1239.842),
}

_LUKE_MIN_UM = 0.31
_LUKE_MAX_UM = 5.504

# Boundary where we switch from Luke (NIR, lossless) to Kischkat (MIR, absorbing).
_CROSSOVER_UM = 1.54


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class SiNDispersion:
    """Callable SiN dispersion model combining Luke (NIR) and Kischkat (MIR)."""

    nir_min_um: float = _LUKE_MIN_UM
    nir_max_um: float = _LUKE_MAX_UM
    crossover_um: float = _CROSSOVER_UM

    def __post_init__(self) -> None:
        self._mir_wl, self._mir_n, self._mir_k = _load_kischkat()

    # ------------------------------------------------------------------

    def n(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Real part of the refractive index at one or many wavelengths."""
        wl_um = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) / 1000.0
        n_vals = np.zeros_like(wl_um, dtype=float)

        nir = wl_um < self.crossover_um
        mir = ~nir

        if nir.any():
            n_vals[nir] = _luke_n(wl_um[nir])
        if mir.any():
            n_vals[mir] = np.interp(
                wl_um[mir], self._mir_wl, self._mir_n,
                left=self._mir_n[0], right=self._mir_n[-1],
            )
        return n_vals.reshape(np.shape(wavelength_nm))

    def k(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Imaginary part of the refractive index (extinction coefficient)."""
        wl_um = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) / 1000.0
        k_vals = np.zeros_like(wl_um, dtype=float)

        mir = wl_um >= self.crossover_um
        if mir.any():
            k_vals[mir] = np.interp(
                wl_um[mir], self._mir_wl, self._mir_k,
                left=self._mir_k[0], right=self._mir_k[-1],
            )
        # Luke region: k ≈ 0
        return k_vals.reshape(np.shape(wavelength_nm))

    def nk(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Complex refractive index n + i k."""
        return self.n(wavelength_nm) + 1j * self.k(wavelength_nm)

    def epsilon(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Complex permittivity ε = (n + i k)²."""
        return self.nk(wavelength_nm) ** 2


# ---------------------------------------------------------------------------
# Internal helpers (defined before the module-level singleton)
# ---------------------------------------------------------------------------


def _luke_n(wl_um: np.ndarray) -> np.ndarray:
    """Evaluate the Luke Sellmeier formula for real n (lossless)."""
    wl2 = wl_um ** 2
    n2 = 1.0
    for B, C in zip(_LUKE_SELLMEIER["B"], _LUKE_SELLMEIER["C"]):
        n2 = n2 + B * wl2 / (wl2 - C ** 2)
    return np.sqrt(np.clip(n2, 1.0, None))


def _load_kischkat() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the tabulated Kischkat 2012 MIR data from CSV."""
    rows = []
    with open(_KISCHKAT_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Module-level singleton + public functions
# ---------------------------------------------------------------------------


_DEFAULT_SIN = SiNDispersion()


def sin_refractive_index(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Complex refractive index of SiN at the given wavelength(s) in nm."""
    return _DEFAULT_SIN.nk(wavelength_nm)


def sin_permittivity(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Complex permittivity of SiN at the given wavelength(s) in nm."""
    return _DEFAULT_SIN.epsilon(wavelength_nm)
