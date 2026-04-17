"""Minimal RCWA-backed demo.

Runs a very short Stage 1 optimization using the real grcwa-based
:class:`RCWASolver` with the public SiN dispersion (Luke NIR + Kischkat
MIR). The iteration counts are shrunk because each grcwa call at
``nG=31, 64x64`` resolution costs ~1 s per wavelength, and every trial
evaluates ~60 wavelengths (30 NIR + 30 MIR). A full 10-iteration run
therefore takes a few minutes.

Run:

    python3 scripts/demo_rcwa.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from lightsail.experiments.main import run_experiment


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "stage1_triangular.yaml"
)
OUTPUT_ROOT = Path("results")


DEMO_OVERRIDES = {
    "name": "demo_rcwa_triangular",
    "solver": {
        "kind": "rcwa",
        "nG": 31,
        "grid_nx": 64,
        "grid_ny": 64,
        "polarization": "average",
    },
    "mobo": {
        "n_init": 6,
        "n_iterations": 4,
        "acqf_num_restarts": 2,
        "acqf_raw_samples": 16,
        "seed": 42,
    },
    # NIR: 12 points instead of 30 to keep demo fast
    "objectives": {
        "nir_reflectance": {"n_points": 12},
        "mir_emissivity": {"n_points": 12},
    },
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("Running RCWA demo (Stage 1, triangular, grcwa + SiN dispersion)")
    print(f"  config   : {CONFIG_PATH}")
    print(f"  overrides: {DEMO_OVERRIDES}")
    print("  Expected runtime: ~1–3 minutes on CPU.\n")

    output_dir = run_experiment(
        config_path=CONFIG_PATH,
        output_root=OUTPUT_ROOT,
        overrides=DEMO_OVERRIDES,
    )

    print("\n" + "=" * 72)
    print("RCWA demo complete. Inspect the artifacts here:")
    print(f"  {output_dir}")
    for rel in [
        "summary.txt",
        "best_design.yaml",
        "plots/pareto_nir_reflectance_vs_mir_emissivity.png",
        "plots/optimization_history.png",
        "plots/spectrum_best.png",
        "plots/structure_topview.png",
    ]:
        target = output_dir / rel
        marker = "✓" if target.exists() else "✗"
        print(f"  {marker} {rel}")


if __name__ == "__main__":
    main()
