"""Aggregate the triangular ensemble runs and pick the absolute best.

Scans ``results/`` for the three ``stage1_triangular_ensemble_s<seed>``
runs, extracts the feasible-only best per seed + per objective, and
compares them to the previous single-seed baseline production run.

Run:

    python3 scripts/analyze_triangular_ensemble.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


SEEDS = [42, 123, 456]
PENALTY_FEASIBLE = 0.01


def _latest_run(tag: str) -> Optional[Path]:
    candidates = sorted(
        (p for p in Path("results").iterdir() if p.is_dir() and tag in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load(run_dir: Path) -> dict:
    return json.loads((run_dir / "trials.json").read_text())


def _feasible(trials: list) -> list:
    return [t for t in trials if t["constraint_penalty"] < PENALTY_FEASIBLE]


def _best(feasible: list, key: str) -> dict:
    return max(feasible, key=lambda t: t["objective_values"][key])


def _params(trial: dict, names: list) -> dict:
    return {name: float(trial["params"][i]) for i, name in enumerate(names)}


def main() -> None:
    runs: dict = {}

    # Load all ensemble seeds.
    for seed in SEEDS:
        tag = f"stage1_triangular_ensemble_s{seed}"
        d = _latest_run(tag)
        if d is None:
            print(f"  ⚠ no run found for seed {seed} (tag={tag})")
            continue
        data = _load(d)
        feas = _feasible(data["trials"])
        runs[seed] = {
            "dir": d,
            "data": data,
            "feasible": feas,
            "n_feasible": len(feas),
            "param_names": data["search_space_names"],
        }
        print(f"  loaded seed {seed}: {d.name}  ({len(feas)} feasible)")

    # Load previous production baseline for comparison.
    baseline_dir = _latest_run("stage1_triangular_production")
    baseline = None
    if baseline_dir is not None:
        bdata = _load(baseline_dir)
        bfeas = _feasible(bdata["trials"])
        baseline = {
            "dir": baseline_dir,
            "data": bdata,
            "feasible": bfeas,
            "param_names": bdata["search_space_names"],
        }
        print(f"\n  baseline (single seed 42): {baseline_dir.name}")

    if not runs:
        print("No ensemble runs found. Aborting.")
        return

    # ------------------------------------------------------------------
    # Per-seed best NIR and best MIR
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Triangular ensemble — per-seed feasible bests")
    print("=" * 80)
    print(f"{'seed':>6}  {'n_feas':>6}  {'NIR best':>10}  {'MIR@NIR':>10}  {'MIR best':>10}")
    print("-" * 60)

    absolute_best_nir = None
    absolute_best_mir = None
    best_nir_per_seed = {}
    best_mir_per_seed = {}

    for seed in SEEDS:
        if seed not in runs:
            continue
        info = runs[seed]
        if not info["feasible"]:
            print(f"{seed:>6}  {'(none)':>6}")
            continue
        b_nir = _best(info["feasible"], "nir_reflectance")
        b_mir = _best(info["feasible"], "mir_emissivity")
        best_nir_per_seed[seed] = b_nir
        best_mir_per_seed[seed] = b_mir

        nir_val = b_nir["objective_values"]["nir_reflectance"]
        mir_at_nir = b_nir["objective_values"]["mir_emissivity"]
        mir_val = b_mir["objective_values"]["mir_emissivity"]
        print(
            f"{seed:>6}  {info['n_feasible']:>6}  "
            f"{nir_val:>10.4f}  {mir_at_nir:>10.4f}  {mir_val:>10.4f}"
        )

        if absolute_best_nir is None or nir_val > absolute_best_nir["objective_values"]["nir_reflectance"]:
            absolute_best_nir = b_nir
            absolute_best_nir["_seed"] = seed
        if absolute_best_mir is None or mir_val > absolute_best_mir["objective_values"]["mir_emissivity"]:
            absolute_best_mir = b_mir
            absolute_best_mir["_seed"] = seed

    # Baseline comparison
    if baseline is not None and baseline["feasible"]:
        b_nir = _best(baseline["feasible"], "nir_reflectance")
        b_mir = _best(baseline["feasible"], "mir_emissivity")
        print(
            f"{'base':>6}  {len(baseline['feasible']):>6}  "
            f"{b_nir['objective_values']['nir_reflectance']:>10.4f}  "
            f"{b_nir['objective_values']['mir_emissivity']:>10.4f}  "
            f"{b_mir['objective_values']['mir_emissivity']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Absolute best across all seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ABSOLUTE BEST across ensemble (triangular, 3 seeds × 180 trials)")
    print("=" * 80)

    if absolute_best_nir is not None:
        seed = absolute_best_nir["_seed"]
        info = runs[seed]
        p = _params(absolute_best_nir, info["param_names"])
        v = absolute_best_nir["objective_values"]
        print(f"\nBest NIR R  —  seed {seed}, trial {absolute_best_nir['trial_id']}")
        print(f"  NIR R       : {v['nir_reflectance']:.4f}")
        print(f"  MIR ε       : {v['mir_emissivity']:.4f}")
        print(f"  fab_penalty : {v['fabrication_penalty']:.4f}")
        print(f"  thickness   : {p['thickness_nm']:.0f} nm")
        print(f"  period      : {p['lattice_period_nm']:.0f} nm")
        print(
            f"  hole (a,b)  : ({p['hole_a_rel']*p['lattice_period_nm']:.0f}, "
            f"{p['hole_b_rel']*p['lattice_period_nm']:.0f}) nm "
            f"(rel {p['hole_a_rel']:.3f}/{p['hole_b_rel']:.3f})"
        )
        print(f"  rotation    : {p['hole_rotation_deg']:.1f}°")
        print(f"  rounding    : {p['corner_rounding']:.2f}")
        print(f"  n_sides     : {int(round(p['shape_parameter']))}")

    if absolute_best_mir is not None:
        seed = absolute_best_mir["_seed"]
        info = runs[seed]
        p = _params(absolute_best_mir, info["param_names"])
        v = absolute_best_mir["objective_values"]
        print(f"\nBest MIR ε  —  seed {seed}, trial {absolute_best_mir['trial_id']}")
        print(f"  NIR R       : {v['nir_reflectance']:.4f}")
        print(f"  MIR ε       : {v['mir_emissivity']:.4f}")
        print(f"  thickness   : {p['thickness_nm']:.0f} nm")
        print(f"  period      : {p['lattice_period_nm']:.0f} nm")

    # Gain vs baseline
    if baseline is not None and baseline["feasible"] and absolute_best_nir:
        b_nir_baseline = _best(baseline["feasible"], "nir_reflectance")["objective_values"]["nir_reflectance"]
        gain_nir = absolute_best_nir["objective_values"]["nir_reflectance"] - b_nir_baseline
        print(f"\nGain vs baseline (single seed 42): NIR R {b_nir_baseline:.4f} → "
              f"{absolute_best_nir['objective_values']['nir_reflectance']:.4f} ({gain_nir:+.4f})")

    # Save YAML summary
    summary = {
        "absolute_best_nir": {
            "seed": int(absolute_best_nir.get("_seed", -1)),
            "trial_id": int(absolute_best_nir["trial_id"]),
            "objectives": {k: float(v) for k, v in absolute_best_nir["objective_values"].items()},
        },
        "absolute_best_mir": {
            "seed": int(absolute_best_mir.get("_seed", -1)),
            "trial_id": int(absolute_best_mir["trial_id"]),
            "objectives": {k: float(v) for k, v in absolute_best_mir["objective_values"].items()},
        },
        "per_seed_bests": {
            int(seed): {
                "n_feasible": info["n_feasible"],
                "best_nir": float(best_nir_per_seed[seed]["objective_values"]["nir_reflectance"])
                if seed in best_nir_per_seed else None,
                "best_mir": float(best_mir_per_seed[seed]["objective_values"]["mir_emissivity"])
                if seed in best_mir_per_seed else None,
            }
            for seed, info in runs.items()
        },
    }
    out_yaml = Path("results") / "triangular_ensemble_summary.yaml"
    out_yaml.write_text(yaml.dump(summary, sort_keys=False))
    print(f"\n  saved summary: {out_yaml}")

    # ------------------------------------------------------------------
    # Joint scatter plot: all feasible points per seed + baseline
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {42: "tab:blue", 123: "tab:orange", 456: "tab:green"}
    for seed in SEEDS:
        if seed not in runs:
            continue
        feas = runs[seed]["feasible"]
        x = [t["objective_values"]["nir_reflectance"] for t in feas]
        y = [t["objective_values"]["mir_emissivity"] for t in feas]
        ax.scatter(
            x, y, c=colors[seed], s=25, alpha=0.45, label=f"seed {seed} (n={len(feas)})"
        )
        # Best NIR as star
        bn = best_nir_per_seed.get(seed)
        if bn:
            ax.scatter(
                bn["objective_values"]["nir_reflectance"],
                bn["objective_values"]["mir_emissivity"],
                c=colors[seed],
                s=240,
                marker="*",
                edgecolors="black",
                linewidths=1.3,
                zorder=5,
            )

    if baseline and baseline["feasible"]:
        feas_base = baseline["feasible"]
        x = [t["objective_values"]["nir_reflectance"] for t in feas_base]
        y = [t["objective_values"]["mir_emissivity"] for t in feas_base]
        ax.scatter(x, y, c="gray", s=20, alpha=0.35, label=f"baseline seed=42 (n={len(feas_base)})")

    ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="NIR target 0.8")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="MIR target 0.5")

    ax.set_xlabel("NIR reflectance")
    ax.set_ylabel("MIR emissivity")
    ax.set_title("Triangular ensemble — 3 seeds × 180 trials (feasible only)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    out_png = Path("results") / "triangular_ensemble_scatter.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  saved scatter: {out_png}")


if __name__ == "__main__":
    main()
