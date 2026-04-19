"""Phase 4-D — Paper-ready figure generation (Nat Commun submission).

Produces 5 main figures from existing Phase 1-4 result CSVs and NPZs:

  Fig 1: 3-zone architecture schematic + key numbers
  Fig 2: Stability ablation (5x3 curvature x ring matrix from Phase 2)
  Fig 3: PASS trajectory (curved + BO ring at t=280)
  Fig 4: Damping enhancement vs ring period (from P3.2)
  Fig 5: Underlayer thermal Pareto + spectrum (from Phase 4-A)

Each figure is publication-quality (300 DPI, sans-serif fonts, panel
labels). Combined into a single REPORT-style PNG too.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def latest_dir(prefix: str) -> Path | None:
    base = Path("results")
    if not base.exists():
        return None
    matches = sorted(base.glob(f"*{prefix}*"))
    return matches[-1] if matches else None


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fig1_architecture(out_dir: Path):
    """3-zone schematic + performance summary."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Schematic
    ax = axes[0]
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) 3-zone sail architecture", loc="left", weight="bold")
    # Sail outline (parabolic edge)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(1.5 * np.cos(th), 1.5 * np.sin(th), facecolor="#cce", edgecolor="black", linewidth=1.2)
    # Center PhC
    ax.fill(1.45 * np.cos(th), 1.45 * np.sin(th), facecolor="#aac", edgecolor="black", linewidth=0.8)
    ax.text(0, 0, "Center PhC\nDesign A\nT=20.7 min", ha="center", va="center", fontsize=10, weight="bold")
    # Outer ring annotation
    ax.text(1.75, 0, "Outer ring\nP=1424 nm\nbeam steering", ha="center", fontsize=8, color="darkblue")
    ax.annotate("", xy=(1.5, 0), xytext=(1.6, 0),
                arrowprops=dict(arrowstyle="<->", color="darkblue"))
    # Curvature
    ax.text(0, -1.85, r"Paraboloid R$_c$ = 30 m", ha="center", fontsize=9, color="darkred")
    # Laser
    ax.annotate("Laser", xy=(0, 1.7), xytext=(0, 2.0),
                arrowprops=dict(arrowstyle="->", color="orange", lw=2),
                ha="center", fontsize=10, color="orange", weight="bold")

    # Performance table
    ax = axes[1]
    ax.axis("off")
    ax.set_title("(b) Mission performance", loc="left", weight="bold")
    rows = [
        ("Acceleration time T", "20.75 min", "-16% vs Norder 2025"),
        ("Distance D to β=0.2", "45.9 Gm", "-12%"),
        ("Sail mass (10 m²)", "5.0 g", "+28% (acceptable)"),
        ("max |xy| (5 s sim)", "0.071 m", "1.8 m limit"),
        ("max |tilt|", "3.15°", "10° limit"),
        ("Spin drift", "0%", "5% limit"),
        ("Thermal T_steady @ 1 GW/m²", "1900 K", "SiN melt 2173 K"),
        ("ε_MIR (with absorber)", "0.164", "+50% vs bare"),
    ]
    cell_text = [[r[1], r[2]] for r in rows]
    row_labels = [r[0] for r in rows]
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=["Value", "Notes"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    fig.tight_layout()
    fig.savefig(out_dir / "Fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig1 → {out_dir/'Fig1_architecture.png'}")


def fig2_ablation_matrix(out_dir: Path):
    """Recycle the t=280 sweep matrix plot."""
    src = Path("results") / "_paper_plots_bundle" / "04b_ablation_matrix_t280.png"
    if not src.exists():
        # Fallback: latest curvature sweep
        d = latest_dir("p4_curvature_sweep")
        if d is None:
            return
        src = d / "curvature_sweep_matrix.png"
        if not src.exists():
            return
    import shutil
    shutil.copy(src, out_dir / "Fig2_ablation.png")
    print(f"  Fig2 → {out_dir/'Fig2_ablation.png'}")


def fig3_trajectory(out_dir: Path):
    """Recycle PASS trajectory (curved + BO ring)."""
    src = Path("results") / "_paper_plots_bundle" / "03b_PASS_curved_BOring_t280.png"
    if not src.exists():
        d = latest_dir("p4_traj_bo_best_Rc30m")
        if d is None:
            return
        src = d / "trajectory.png"
    import shutil
    shutil.copy(src, out_dir / "Fig3_PASS_trajectory.png")
    print(f"  Fig3 → {out_dir/'Fig3_PASS_trajectory.png'}")


def fig4_damping(out_dir: Path):
    src = Path("results") / "_paper_plots_bundle" / "05_damping_enhancement.png"
    if not src.exists():
        return
    import shutil
    shutil.copy(src, out_dir / "Fig4_damping.png")
    print(f"  Fig4 → {out_dir/'Fig4_damping.png'}")


def fig5_thermal(out_dir: Path):
    """MIR absorber Pareto + spectrum."""
    p4_a3 = latest_dir("p4_multi_material")
    if p4_a3 is None:
        return
    rows = load_csv(p4_a3 / "multi_material_scan.csv")
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (a) ε_MIR vs mass Pareto
    ax = axes[0]
    eps_mir = [float(r["mean_eps_MIR"]) for r in rows]
    mass = [float(r["extra_g_per_m2"]) for r in rows]
    R_nir = [float(r["mean_R_NIR"]) for r in rows]
    sc = ax.scatter(mass, eps_mir, c=R_nir, cmap="RdYlGn", s=80, edgecolors="black")
    ax.set_xlabel("Extra mass [g/m²]")
    ax.set_ylabel(r"$\epsilon_{MIR}$ avg (8–14 µm)")
    ax.set_title("(a) MIR absorber design Pareto", loc="left", weight="bold")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, label=r"$R_{NIR}$")

    # (b) Thermal balance vs intensity
    ax = axes[1]
    sigma_SB = 5.670374419e-8
    alpha_NIR = 0.0002
    eps_MIR_best = 0.164
    eps_MIR_bare = 0.108
    I_GW = np.logspace(-1, 1.5, 30)
    T_best = (alpha_NIR * I_GW * 1e9 / (2 * sigma_SB * eps_MIR_best))**0.25
    T_bare = (alpha_NIR * I_GW * 1e9 / (2 * sigma_SB * eps_MIR_bare))**0.25
    ax.plot(I_GW, T_bare, "--", color="gray", label=r"bare PhC ($\epsilon_{MIR}=0.108$)")
    ax.plot(I_GW, T_best, "-", color="C2", linewidth=2, label=r"+ MIR absorber ($\epsilon_{MIR}=0.164$)")
    ax.axhline(2173, color="r", linestyle=":", label="SiN melt 2173 K")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(1.1, 100, "Gieseler 2024\n1 GW/m²", fontsize=8)
    ax.axvline(10.0, color="orange", linestyle=":", alpha=0.5)
    ax.text(11, 100, "Starshot\n10 GW/m²", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Laser intensity $I_0$ [GW/m²]")
    ax.set_ylabel(r"$T_{steady}$ [K]")
    ax.set_title("(b) Thermal balance (sail rest frame)", loc="left", weight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "Fig5_thermal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig5 → {out_dir/'Fig5_thermal.png'}")


def main():
    if not HAS_MPL:
        print("matplotlib required.")
        return
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating paper-ready figures ...")
    fig1_architecture(out_dir)
    fig2_ablation_matrix(out_dir)
    fig3_trajectory(out_dir)
    fig4_damping(out_dir)
    fig5_thermal(out_dir)

    print(f"\nAll figures in {out_dir}/")


if __name__ == "__main__":
    main()
