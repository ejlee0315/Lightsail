"""Post-production analysis for the latest Stage 1 run.

Does two things:

1. **Extracts feasible-only best designs** from trials.json. The default
   ``best_design.yaml`` uses penalty mode which can point to infeasible
   (high-penalty) trials as "best". This script separates feasible trials
   (penalty < threshold) and saves the top-K per objective to
   ``feasible_bests.yaml``.

2. **Validates top feasible Pareto candidates at higher nG (81)**. The main
   BO loop ran at nG=41 for speed; this re-evaluates the top 5 feasible
   designs at nG=81 and reports the Δ so we can tell whether the ranking
   is robust under higher Fourier truncation.

Usage:

    python3 scripts/validate_top_designs.py <run_dir>
    python3 scripts/validate_top_designs.py   # uses latest results/ dir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver


PENALTY_FEASIBLE = 0.01
TOP_K = 5


def _latest_run_dir() -> Path:
    roots = sorted(
        Path("results").iterdir(),
        key=lambda p: p.stat().st_mtime,
    )
    return roots[-1]


def _load_trials(run_dir: Path) -> tuple[list[dict], list[str], dict]:
    data = json.loads((run_dir / "trials.json").read_text())
    return data["trials"], data["search_space_names"], data["objective_targets"]


def _trial_params(trial: dict, names: list[str]) -> dict[str, float]:
    return {name: float(trial["params"][i]) for i, name in enumerate(names)}


def _summarize_design(trial: dict, names: list[str]) -> dict:
    return {
        "trial_id": int(trial["trial_id"]),
        "iteration": int(trial["iteration"]),
        "source": trial["source"],
        "constraint_penalty": float(trial["constraint_penalty"]),
        "objectives": {
            k: float(v) for k, v in trial["objective_values"].items()
        },
        "params": _trial_params(trial, names),
    }


def _phc_from_params(params: dict[str, float]) -> PhCReflector:
    """Rebuild a PhCReflector from a flat param dict.

    Accepts either the new relative parameterization (``hole_a_rel``) or
    the legacy absolute one (``hole_a_nm``). This lets us re-validate old
    production runs as well as the new ones.
    """
    period = float(params["lattice_period_nm"])
    if "hole_a_rel" in params:
        a_rel = float(params["hole_a_rel"])
    else:
        a_rel = float(params["hole_a_nm"]) / period
    if "hole_b_rel" in params:
        b_rel = float(params["hole_b_rel"])
    else:
        b_rel = float(params["hole_b_nm"]) / period

    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        n_rings=6,
        thickness_nm=float(params["thickness_nm"]),
        lattice_period_nm=period,
        hole_a_rel=a_rel,
        hole_b_rel=b_rel,
        hole_rotation_deg=float(params["hole_rotation_deg"]),
        corner_rounding=float(params["corner_rounding"]),
        shape_parameter=float(params["shape_parameter"]),
    )


def main() -> None:
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = _latest_run_dir()

    print(f"Analyzing run: {run_dir}")
    trials, param_names, obj_targets = _load_trials(run_dir)
    print(f"  total trials: {len(trials)}")

    feasible = [t for t in trials if t["constraint_penalty"] < PENALTY_FEASIBLE]
    print(f"  strictly feasible (<{PENALTY_FEASIBLE}): {len(feasible)}")

    # ------------------------------------------------------------------
    # 1. Feasible-only bests per objective
    # ------------------------------------------------------------------
    best_nir = max(
        feasible, key=lambda t: t["objective_values"]["nir_reflectance"]
    )
    best_mir = max(
        feasible, key=lambda t: t["objective_values"]["mir_emissivity"]
    )

    # Balanced: NIR + MIR, ignoring fab (which is ~0 for feasible designs)
    def balanced_score(t):
        v = t["objective_values"]
        return v["nir_reflectance"] + v["mir_emissivity"]

    best_balanced = max(feasible, key=balanced_score)

    feasible_bests = {
        "run_dir": str(run_dir),
        "penalty_threshold": PENALTY_FEASIBLE,
        "n_feasible": len(feasible),
        "best_nir_reflectance": _summarize_design(best_nir, param_names),
        "best_mir_emissivity": _summarize_design(best_mir, param_names),
        "best_balanced": _summarize_design(best_balanced, param_names),
    }
    out_yaml = run_dir / "feasible_bests.yaml"
    out_yaml.write_text(yaml.dump(feasible_bests, sort_keys=False))
    print(f"  saved: {out_yaml}")

    # ------------------------------------------------------------------
    # 2. Pick top-K feasible Pareto candidates for nG=81 validation
    # ------------------------------------------------------------------
    # Candidate pool: feasible trials whose trial_id appears in Pareto
    # OR whose NIR/MIR rank is top-5.
    data_full = json.loads((run_dir / "trials.json").read_text())
    pareto_ids = set(int(i) for i in data_full["pareto_indices"])
    feasible_pareto = [t for t in feasible if t["trial_id"] in pareto_ids]
    # Fall back to top-K by NIR if the feasible Pareto set is too small.
    if len(feasible_pareto) < TOP_K:
        feasible_pareto = sorted(
            feasible,
            key=lambda t: -t["objective_values"]["nir_reflectance"],
        )[:TOP_K]
    else:
        feasible_pareto = sorted(
            feasible_pareto,
            key=lambda t: -t["objective_values"]["nir_reflectance"],
        )[:TOP_K]

    print(
        f"\nValidating top {len(feasible_pareto)} feasible designs at nG=81..."
    )
    solver_hi = RCWASolver(
        config=RCWAConfig(nG=81, grid_nx=96, grid_ny=96, polarization="average")
    )

    nir_band = np.linspace(1350.0, 1650.0, 30)
    mir_band = np.linspace(8000.0, 14000.0, 30)

    print(
        f"{'trial':>6}  {'NIR@41':>8}  {'NIR@81':>8}  {'ΔNIR':>8}  "
        f"{'MIR@41':>8}  {'MIR@81':>8}  {'ΔMIR':>8}"
    )
    print("-" * 70)

    validation_rows = []
    for trial in feasible_pareto:
        params = _trial_params(trial, param_names)
        phc = _phc_from_params(params)
        structure = phc.to_structure()

        # Re-evaluate at high nG
        R_hi = solver_hi.evaluate_reflectivity(structure, nir_band)
        T_hi = solver_hi.evaluate_transmission(structure, nir_band)
        eps_mir = solver_hi.evaluate_emissivity(structure, mir_band)
        nir_hi = float(R_hi.mean() * 0.7 + R_hi.min() * 0.3)  # mean+min mixture
        mir_hi = float(eps_mir.mean())

        nir_lo = trial["objective_values"]["nir_reflectance"]
        mir_lo = trial["objective_values"]["mir_emissivity"]

        print(
            f"{trial['trial_id']:>6d}  {nir_lo:>8.4f}  {nir_hi:>8.4f}  "
            f"{nir_hi - nir_lo:>+8.4f}  {mir_lo:>8.4f}  {mir_hi:>8.4f}  "
            f"{mir_hi - mir_lo:>+8.4f}"
        )

        validation_rows.append(
            {
                "trial_id": int(trial["trial_id"]),
                "nir_reflectance_nG41": nir_lo,
                "nir_reflectance_nG81": nir_hi,
                "mir_emissivity_nG41": mir_lo,
                "mir_emissivity_nG81": mir_hi,
                "delta_nir": nir_hi - nir_lo,
                "delta_mir": mir_hi - mir_lo,
            }
        )

    # Save validation results
    out_val = run_dir / "nG81_validation.yaml"
    out_val.write_text(yaml.dump({"validated": validation_rows}, sort_keys=False))
    print(f"\n  saved: {out_val}")

    # Ranking stability check
    ranking_lo = [r["trial_id"] for r in validation_rows]
    ranking_hi = [
        r["trial_id"]
        for r in sorted(
            validation_rows, key=lambda r: -r["nir_reflectance_nG81"]
        )
    ]
    if ranking_lo == ranking_hi:
        print("\n  Ranking at nG=41 and nG=81 is CONSISTENT — BO result is trustworthy.")
    else:
        print("\n  ⚠ Ranking changed at nG=81. Top design at nG=41:", ranking_lo[0])
        print("  Top design at nG=81:", ranking_hi[0])
        print("  Consider re-running BO at higher nG if this matters.")


if __name__ == "__main__":
    main()
