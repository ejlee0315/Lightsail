"""Visualization helpers for :class:`RunResult` from the MOBO runner."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lightsail.optimization.mobo_runner import RunResult


# ---------------------------------------------------------------------------
# 2D Pareto scatter
# ---------------------------------------------------------------------------


def plot_pareto_scatter(
    run: RunResult,
    x_objective: str,
    y_objective: str,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """2D scatter of all trials with Pareto-optimal set highlighted.

    Initial-sample points are drawn in grey, BO-proposed points in blue,
    and the non-dominated (Pareto) set in red with black edges.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    xs = np.array([t.objective_values[x_objective] for t in run.trials])
    ys = np.array([t.objective_values[y_objective] for t in run.trials])
    init_mask = np.array([t.source == "init" for t in run.trials], dtype=bool)

    ax.scatter(
        xs[init_mask],
        ys[init_mask],
        c="gray",
        s=30,
        alpha=0.6,
        label="init (Sobol)",
    )
    ax.scatter(
        xs[~init_mask],
        ys[~init_mask],
        c="tab:blue",
        s=40,
        alpha=0.75,
        label="BO",
    )

    if run.pareto_indices:
        px = [run.trials[i].objective_values[x_objective] for i in run.pareto_indices]
        py = [run.trials[i].objective_values[y_objective] for i in run.pareto_indices]
        ax.scatter(
            px,
            py,
            c="tab:red",
            s=110,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="Pareto front",
        )

    x_target = run.objective_targets.get(x_objective, "maximize")
    y_target = run.objective_targets.get(y_objective, "maximize")
    ax.set_xlabel(f"{x_objective} ({'↑' if x_target == 'maximize' else '↓'})")
    ax.set_ylabel(f"{y_objective} ({'↑' if y_target == 'maximize' else '↓'})")
    ax.set_title(title or f"Pareto: {x_objective} vs {y_objective}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Per-objective running-best history
# ---------------------------------------------------------------------------


def plot_optimization_history(
    run: RunResult,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot, for each objective, the running-best value vs trial index.

    Initial-sample region is shaded so the BO start is visible.
    """
    n_obj = len(run.objective_names)
    fig, axes = plt.subplots(
        n_obj, 1, figsize=(10, 2.4 * n_obj), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for ax, name in zip(axes, run.objective_names):
        target = run.objective_targets[name]
        values = np.array([t.objective_values[name] for t in run.trials], dtype=float)

        if target == "maximize":
            running = np.maximum.accumulate(values)
        else:
            running = np.minimum.accumulate(values)

        ax.scatter(np.arange(len(values)), values, c="lightgray", s=12, zorder=2)
        ax.plot(
            running,
            color="tab:blue",
            lw=2,
            zorder=3,
            label=f"running {'max' if target == 'maximize' else 'min'}",
        )
        ax.axvspan(
            -0.5,
            run.config.n_init - 0.5,
            alpha=0.1,
            color="tab:orange",
            label="init phase",
        )
        ax.set_ylabel(f"{name}\n({target})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("trial index")
    fig.suptitle("Optimization history", y=1.0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def summarize_best(run: RunResult, top_k: int = 3) -> str:
    """Return a multi-line text summary of top-K candidates per objective."""
    lines = []
    lines.append(
        f"Best candidates summary  —  n_trials={run.n_trials}, "
        f"pareto={len(run.pareto_indices)}"
    )
    lines.append("=" * 70)

    for name in run.objective_names:
        target = run.objective_targets[name]
        arrow = "↑" if target == "maximize" else "↓"
        lines.append(f"\n{name} ({arrow})")

        sorted_trials = sorted(
            run.trials,
            key=lambda t: float(t.objective_values.get(name, 0.0)),
            reverse=(target == "maximize"),
        )[:top_k]

        for rank, trial in enumerate(sorted_trials, 1):
            feas = "OK " if trial.feasible else "!! "
            lines.append(
                f"  {rank}. {feas}trial {trial.trial_id:3d} [{trial.source:14s}] "
                f"value={trial.objective_values[name]:+.4f}  "
                f"penalty={trial.constraint_penalty:.3f}"
            )
    return "\n".join(lines)
