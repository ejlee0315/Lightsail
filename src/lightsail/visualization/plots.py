"""Visualization utilities for spectra, structures, and optimization results.

All helpers return the ``matplotlib.figure.Figure`` they produce so
callers can embed them in multi-panel figures or save them manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from lightsail.geometry.base import HoleShape, Ring, Structure
from lightsail.optimization.optimizer import ParetoFront
from lightsail.simulation.results import SimulationResult


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def plot_spectrum(
    result: SimulationResult,
    title: str = "Optical Spectrum",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot reflectance, transmittance, and absorptance vs wavelength."""
    fig, ax = plt.subplots(figsize=(10, 5))
    wl_um = result.wavelengths_nm / 1000.0

    ax.plot(wl_um, result.reflectance, label="Reflectance", color="tab:blue")
    ax.plot(wl_um, result.transmittance, label="Transmittance", color="tab:orange")
    ax.plot(
        wl_um,
        result.absorptance,
        label="Absorptance",
        color="tab:red",
        linestyle="--",
    )

    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Broadband NIR + MIR spectrum (two side-by-side subplots)
# ---------------------------------------------------------------------------


def plot_broadband_spectrum(
    result: SimulationResult,
    title: str = "Best design spectrum",
    nir_band_nm: tuple[float, float] = (1000.0, 2500.0),
    mir_band_nm: tuple[float, float] = (5000.0, 15000.0),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot R/T/ε for a single design in two side-by-side subplots (NIR + MIR).

    Useful when the wavelength range spans more than an order of magnitude,
    which makes a single linear axis unreadable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    wl_nm = result.wavelengths_nm

    for ax, band, label in zip(
        axes, [nir_band_nm, mir_band_nm], ["NIR", "MIR"]
    ):
        mask = (wl_nm >= band[0]) & (wl_nm <= band[1])
        if not np.any(mask):
            ax.text(0.5, 0.5, "no samples in band", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{label}  {band[0]/1000:.1f}-{band[1]/1000:.1f} µm")
            continue
        wl_um = wl_nm[mask] / 1000.0
        ax.plot(wl_um, result.reflectance[mask], color="tab:blue", label="R")
        ax.plot(wl_um, result.transmittance[mask], color="tab:orange", label="T")
        ax.plot(
            wl_um,
            result.absorptance[mask],
            color="tab:red",
            linestyle="--",
            label="ε",
        )
        ax.axvspan(band[0] / 1000.0, band[1] / 1000.0, alpha=0.06, color="gray")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_title(f"{label}  {band[0]/1000:.1f}-{band[1]/1000:.1f} µm")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Isolated hole shape (useful for inspecting the rounded-polygon family)
# ---------------------------------------------------------------------------


def plot_hole_shape(
    shape: HoleShape,
    n_pts: int = 256,
    title: Optional[str] = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a single HoleShape boundary in its local frame."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")

    pts = shape.boundary(n_pts) / 1000.0  # nm -> µm
    poly = MplPolygon(
        pts,
        closed=True,
        fill=True,
        facecolor="lightblue",
        edgecolor="tab:blue",
        linewidth=1.5,
    )
    ax.add_patch(poly)

    lim = max(np.abs(pts).max() * 1.2, 1e-6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        title
        or f"HoleShape n={shape.n_sides}, rounding={shape.corner_rounding:.2f}, "
        f"a={shape.a_nm:.0f}, b={shape.b_nm:.0f}"
    )
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Full structure top-view
# ---------------------------------------------------------------------------


def _ring_patch_pair(ring: Ring, n_pts: int = 256) -> tuple[MplPolygon, MplPolygon]:
    inner, outer = ring.boundary(n_pts)
    inner_patch = MplPolygon(
        inner / 1000.0, closed=True, fill=False, edgecolor="tab:red", linewidth=0.8
    )
    outer_patch = MplPolygon(
        outer / 1000.0, closed=True, fill=False, edgecolor="tab:red", linewidth=0.8
    )
    return inner_patch, outer_patch


def plot_structure_topview(
    structure: Structure,
    title: str = "Structure (top view)",
    n_pts_per_shape: int = 96,
    hole_edge_color: str = "tab:blue",
    hole_fill: bool = True,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a top-down view of a Structure, including polygon holes and
    (optionally warped) rings."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # ---- holes ----
    for hole in structure.holes:
        pts = hole.boundary_global(n_pts_per_shape) / 1000.0
        poly = MplPolygon(
            pts,
            closed=True,
            fill=hole_fill,
            facecolor="lightblue" if hole_fill else "none",
            edgecolor=hole_edge_color,
            linewidth=0.6,
        )
        ax.add_patch(poly)

    # ---- rings ----
    for ring in structure.rings:
        inner_patch, outer_patch = _ring_patch_pair(ring, n_pts=256)
        ax.add_patch(inner_patch)
        ax.add_patch(outer_patch)

    if structure.extent_nm:
        lim = structure.extent_nm / 1000.0 * 0.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


def plot_pareto_front(
    pareto: ParetoFront,
    x_objective: str,
    y_objective: str,
    title: str = "Pareto Front",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """2D scatter of Pareto-optimal solutions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = [t.objective_values.get(x_objective, 0.0) for t in pareto.trials]
    y = [t.objective_values.get(y_objective, 0.0) for t in pareto.trials]

    ax.scatter(x, y, c="tab:green", s=60, edgecolors="black", zorder=5)
    ax.set_xlabel(x_objective)
    ax.set_ylabel(y_objective)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
