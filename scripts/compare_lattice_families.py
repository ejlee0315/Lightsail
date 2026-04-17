"""Compare Stage 1 production runs across lattice families.

Scans ``results/`` for the latest ``stage1_<family>_production`` directory
per family, extracts the feasible-only best designs, and prints a
side-by-side comparison table plus a joint NIR-vs-MIR scatter plot.

Usage:

    python3 scripts/compare_lattice_families.py

Optional: pass specific run directories as positional arguments to pin
exact runs instead of picking the latest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


FAMILIES = ["triangular", "hexagonal", "rectangular", "pentagonal"]
PENALTY_FEASIBLE = 0.01


def _latest_run_for(family: str, results_dir: Path = Path("results")) -> Optional[Path]:
    tag = f"stage1_{family}_production"
    candidates = sorted(
        (p for p in results_dir.iterdir() if p.is_dir() and tag in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load_run(run_dir: Path) -> dict:
    with open(run_dir / "trials.json") as f:
        data = json.load(f)
    return data


def _feasible_bests(data: dict) -> dict:
    trials = data["trials"]
    feasible = [t for t in trials if t["constraint_penalty"] < PENALTY_FEASIBLE]
    if not feasible:
        return {"n_feasible": 0}
    best_nir = max(feasible, key=lambda t: t["objective_values"]["nir_reflectance"])
    best_mir = max(feasible, key=lambda t: t["objective_values"]["mir_emissivity"])
    best_bal = max(
        feasible,
        key=lambda t: (
            t["objective_values"]["nir_reflectance"]
            + t["objective_values"]["mir_emissivity"]
        ),
    )
    return {
        "n_feasible": len(feasible),
        "best_nir": best_nir,
        "best_mir": best_mir,
        "best_balanced": best_bal,
        "all_feasible": feasible,
    }


def _fmt_trial(trial: dict) -> str:
    v = trial["objective_values"]
    return (
        f"NIR={v['nir_reflectance']:.4f}  MIR={v['mir_emissivity']:.4f}  "
        f"[trial {trial['trial_id']:3d}]"
    )


def _print_comparison_table(runs: dict) -> None:
    print("=" * 88)
    print("Stage 1 lattice family comparison (feasible-only)")
    print("=" * 88)
    header = f"{'family':12s}  {'n_feas':>6}  {'best NIR R':>28}  {'best MIR ε':>28}"
    print(header)
    print("-" * 88)
    for fam in FAMILIES:
        info = runs[fam]
        if info is None:
            print(f"{fam:12s}  (no run found)")
            continue
        bests = info["bests"]
        if bests["n_feasible"] == 0:
            print(f"{fam:12s}  {0:>6d}  (no feasible designs)")
            continue
        nir = bests["best_nir"]["objective_values"]
        mir = bests["best_mir"]["objective_values"]
        print(
            f"{fam:12s}  {bests['n_feasible']:>6d}  "
            f"NIR={nir['nir_reflectance']:.4f} MIR={nir['mir_emissivity']:.4f}    "
            f"NIR={mir['nir_reflectance']:.4f} MIR={mir['mir_emissivity']:.4f}"
        )
    print()


def _print_best_designs(runs: dict) -> None:
    print("=" * 88)
    print("Best NIR winner per family (params)")
    print("=" * 88)
    for fam in FAMILIES:
        info = runs[fam]
        if info is None or info["bests"]["n_feasible"] == 0:
            continue
        bests = info["bests"]
        trial = bests["best_nir"]
        v = trial["objective_values"]
        p = _params_dict(trial, info["param_names"])
        print(f"\n  {fam.upper()}  (trial {trial['trial_id']}, {trial['source']})")
        print(f"    NIR R              : {v['nir_reflectance']:.4f}")
        print(f"    MIR ε              : {v['mir_emissivity']:.4f}")
        print(f"    fabrication_penalty: {v['fabrication_penalty']:.4f}")
        print(f"    thickness_nm       : {p['thickness_nm']:.0f}")
        print(f"    lattice_period_nm  : {p['lattice_period_nm']:.0f}")
        a_rel = p.get("hole_a_rel", 0.0)
        b_rel = p.get("hole_b_rel", 0.0)
        period = p["lattice_period_nm"]
        print(
            f"    hole (a,b)         : ({a_rel*period:.0f}, {b_rel*period:.0f}) nm "
            f"(rel {a_rel:.3f}/{b_rel:.3f})"
        )
        print(f"    hole_rotation_deg  : {p['hole_rotation_deg']:.1f}°")
        print(f"    corner_rounding    : {p['corner_rounding']:.2f}")
        print(f"    n_sides (shape)    : {int(round(p['shape_parameter']))}")
    print()


def _params_dict(trial: dict, param_names: list) -> dict:
    return {name: trial["params"][i] for i, name in enumerate(param_names)}


def _joint_scatter(runs: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "triangular": "tab:blue",
        "hexagonal": "tab:orange",
        "rectangular": "tab:purple",
        "pentagonal": "tab:green",
    }

    for fam in FAMILIES:
        info = runs[fam]
        if info is None or info["bests"]["n_feasible"] == 0:
            continue
        feas = info["bests"]["all_feasible"]
        x = [t["objective_values"]["nir_reflectance"] for t in feas]
        y = [t["objective_values"]["mir_emissivity"] for t in feas]
        ax.scatter(x, y, c=colors[fam], s=30, alpha=0.5, label=f"{fam} (all feasible)")

        # Highlight best NIR + best MIR
        nir_best = info["bests"]["best_nir"]["objective_values"]
        mir_best = info["bests"]["best_mir"]["objective_values"]
        ax.scatter(
            nir_best["nir_reflectance"],
            nir_best["mir_emissivity"],
            c=colors[fam],
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label=f"{fam} best NIR",
        )
        ax.scatter(
            mir_best["nir_reflectance"],
            mir_best["mir_emissivity"],
            c=colors[fam],
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label=f"{fam} best MIR",
        )

    # Target lines
    ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="NIR target R=0.8")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="MIR target ε=0.5")

    ax.set_xlabel("NIR reflectance")
    ax.set_ylabel("MIR emissivity")
    ax.set_title("Lattice family comparison — feasible-only designs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Joint scatter saved: {save_path}")


def main() -> None:
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = []
        for fam in FAMILIES:
            d = _latest_run_for(fam)
            if d is not None:
                run_dirs.append(d)

    runs: dict = {}
    for fam in FAMILIES:
        # Pick the run dir matching this family
        matching = [d for d in run_dirs if f"stage1_{fam}_production" in d.name]
        if not matching:
            runs[fam] = None
            continue
        d = matching[-1]
        data = _load_run(d)
        runs[fam] = {
            "run_dir": d,
            "param_names": data["search_space_names"],
            "bests": _feasible_bests(data),
        }
        print(f"  loaded {fam:12s}: {d.name}")
    print()

    _print_comparison_table(runs)
    _print_best_designs(runs)

    # Save joint scatter next to the triangular run (or first available).
    base_dir = None
    for fam in FAMILIES:
        if runs[fam]:
            base_dir = runs[fam]["run_dir"].parent
            break
    if base_dir is not None:
        out_path = base_dir / "lattice_family_comparison.png"
        _joint_scatter(runs, out_path)


if __name__ == "__main__":
    main()
