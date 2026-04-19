"""Collect Stage 2 FMM multi-seed best designs into a comparison table.

For each ``results/<*>_stage2_fmm_s<SEED>/`` directory, reads
``best_design.yaml`` (primary objective best) and the Pareto front,
prints a per-seed summary, and writes a combined CSV/Markdown.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    base = Path("results")
    seeds = (42, 123, 456)
    out_dir = base / f"{datetime.now():%Y-%m-%d_%H%M%S}_stage2_fmm_collected"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("=== Stage 2 FMM multi-seed best designs ===\n")
    for seed in seeds:
        matches = sorted(
            base.glob(f"*_stage2_fmm_s{seed}"), key=lambda p: p.name
        )
        if not matches:
            print(f"  seed {seed}: no run directory found")
            continue
        run_dir = matches[-1]
        best_path = run_dir / "best_design.yaml"
        if not best_path.exists():
            print(f"  seed {seed}: no best_design.yaml in {run_dir.name}")
            continue
        best = yaml.safe_load(best_path.read_text())
        params = best.get("params", {})
        objs = best.get("objectives", {})

        n_pareto = 0
        try:
            pareto_arr = np.load(run_dir / "pareto_indices.npy")
            n_pareto = int(pareto_arr.size)
        except Exception:
            pass

        row = {
            "seed": seed,
            "run_dir": run_dir.name,
            "grating_period_nm": params.get("grating_period_nm"),
            "duty_cycle": params.get("duty_cycle"),
            "curvature": params.get("curvature"),
            "asymmetry": params.get("asymmetry"),
            "ring_width_um": params.get("ring_width_um"),
            "stabilization": objs.get("stabilization"),
            "nir_reflectance": objs.get("nir_reflectance"),
            "mir_emissivity": objs.get("mir_emissivity"),
            "fabrication_penalty": objs.get("fabrication_penalty"),
            "n_pareto": n_pareto,
        }
        rows.append(row)

        print(f"  seed {seed}  ({run_dir.name}):")
        print(
            f"    period={row['grating_period_nm']:.0f} nm  "
            f"duty={row['duty_cycle']:.2f}  curv={row['curvature']:+.3f}  "
            f"asym={row['asymmetry']:+.3f}  ring_w={row['ring_width_um']:.1f} µm"
        )
        print(
            f"    stabilization = {row['stabilization']:.4f}   "
            f"nir = {row['nir_reflectance']:.4f}   "
            f"#Pareto = {row['n_pareto']}"
        )
        print()

    if rows:
        csv_path = out_dir / "stage2_fmm_multi_seed_best.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        md_path = out_dir / "stage2_fmm_multi_seed_best.md"
        with open(md_path, "w") as f:
            f.write("# Stage 2 FMM multi-seed best designs (P5)\n\n")
            f.write(
                "| Seed | Period [nm] | Duty | Curvature | Asymmetry | "
                "Ring width [µm] | stabilization | nir_R | #Pareto |\n"
            )
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(
                    f"| {r['seed']} "
                    f"| {r['grating_period_nm']:.0f} "
                    f"| {r['duty_cycle']:.2f} "
                    f"| {r['curvature']:+.3f} "
                    f"| {r['asymmetry']:+.3f} "
                    f"| {r['ring_width_um']:.1f} "
                    f"| {r['stabilization']:.4f} "
                    f"| {r['nir_reflectance']:.4f} "
                    f"| {r['n_pareto']} |\n"
                )

        print(f"CSV     → {csv_path}")
        print(f"Summary → {md_path}")


if __name__ == "__main__":
    main()
