"""P5 — Regenerate the paper figures from the Phase 1–4 outputs.

Combines results from:
    P2.3   results/.../graphene_scan_summary.csv          MIR Pareto
    P3.2   results/.../damping_enhancement_sweep.csv      Ring damping
    P4     results/.../integrated_sail.csv                3-zone summary
    P5     results/.../ring_tolerance.csv                 Period sensitivity

into a single multi-panel figure suitable for the manuscript:
    Fig X.  (a) functional partition diagram (text-only schematic)
            (b) graphene N-layer Pareto (ε_MIR vs mass)
            (c) damping enhancement ratio vs ring period (per β)
            (d) integrated 3-zone comparison bar chart
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np


def latest_dir(prefix: str) -> Optional[Path]:
    base = Path("results")
    if not base.exists():
        return None
    matches = sorted(p for p in base.iterdir() if prefix in p.name)
    return matches[-1] if matches else None


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; install matplotlib to regenerate figures.")
        return

    out_dir = Path("results") / "paper_figures_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Schematic
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("(a) Functional partition", loc="left", fontsize=11)
    ax.text(
        0.05, 0.85,
        "Central PhC reflector\n  (Design A: T=20.7 min, MFS=380 nm)",
        fontsize=10,
    )
    ax.text(
        0.05, 0.55,
        "Outer metagrating ring\n  (k_θθ stiffness + relativistic damping)",
        fontsize=10,
    )
    ax.text(
        0.05, 0.25,
        "Backside thermal layer\n  (graphene tested → MIR reflective; investigate alternative)",
        fontsize=10, color="firebrick",
    )

    # (b) Graphene Pareto
    ax = axes[0, 1]
    p23 = latest_dir("p2_3_graphene_scan")
    if p23:
        rows = load_csv(p23 / "graphene_scan_summary.csv")
        if rows:
            n = [int(r["n_layers"]) for r in rows]
            eps_mir = [float(r["mean_eps_MIR"]) for r in rows]
            R_nir = [float(r["mean_R_NIR"]) for r in rows]
            ax.plot(n, eps_mir, "o-", label="ε_MIR (target ↑)")
            ax.plot(n, R_nir, "s-", label="R_NIR (preserve)")
            ax.set_xlabel("# graphene monolayers")
            ax.set_ylabel("Band-mean")
            ax.set_title("(b) Graphene N-layer scan (P2.3)", loc="left", fontsize=11)
            ax.legend()
            ax.grid(alpha=0.3)

    # (c) Damping enhancement
    ax = axes[1, 0]
    p32 = latest_dir("p3_2_damping_verify")
    if p32:
        rows = load_csv(p32 / "damping_enhancement_sweep.csv")
        if rows:
            by_beta = {}
            for r in rows:
                by_beta.setdefault(float(r["beta"]), []).append(r)
            for beta in sorted(by_beta):
                periods = [float(r["ring_period_nm"]) for r in by_beta[beta]]
                ratios = [float(r["enhancement_ratio"]) for r in by_beta[beta]]
                # Clip extreme ratios for visual clarity
                ratios = [np.clip(r, -100, 100) for r in ratios]
                ax.plot(periods, ratios, "o-", label=f"β = {beta:.2f}")
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
            ax.set_xlabel("Ring grating period [nm]")
            ax.set_ylabel("(∂C_pr,2/∂θ)_ring / (∂C_pr,2/∂θ)_bare  (clipped ±100)")
            ax.set_title("(c) Outer-ring damping enhancement (P3.2)", loc="left", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    # (d) Integrated 3-zone bar chart
    ax = axes[1, 1]
    p4 = latest_dir("p4_integrated_sail")
    if p4:
        rows = load_csv(p4 / "integrated_sail.csv")
        if rows:
            labels = [r["label"] for r in rows]
            T_min = [float(r["T_minutes"]) for r in rows]
            x = np.arange(len(labels))
            ax.bar(x, T_min, color=["C0", "C1", "C3"])
            for xi, t in zip(x, T_min):
                ax.text(xi, t + 0.3, f"{t:.1f}", ha="center", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace(": ", ":\n") for l in labels], fontsize=9)
            ax.set_ylabel("Acceleration time T [min]")
            ax.set_title("(d) Integrated 3-zone comparison (P4)", loc="left", fontsize=11)
            ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = out_dir / "paper_figure_combined.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Combined paper figure saved to {out_path}")


if __name__ == "__main__":
    main()
