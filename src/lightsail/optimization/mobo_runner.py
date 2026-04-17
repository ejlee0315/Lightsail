"""BoTorch-based multi-objective Bayesian optimization runner.

Architecture
------------
The runner is a single class :class:`MOBORunner` that, given:

- an :class:`ObjectiveEvaluator` (geometry + solver + constraints + objectives)
- a :class:`SearchSpace` (parameter bounds)
- a :class:`MOBOConfig` (BO hyper-parameters)

runs ``n_init`` initial Sobol (or LHS) samples followed by ``n_iterations``
BO iterations of batch size ``batch_size``. Each BO iteration fits a
``SingleTaskGP`` to the current data (normalized inputs, standardized
outputs) and optimizes ``qLogNoisyExpectedHypervolumeImprovement``
to propose the next candidates.

All work happens in a normalized ``[0, 1]^d`` cube. Physical parameter
vectors are obtained by ``SearchSpace.denormalize`` before calling the
evaluator. Objective values from the evaluator are turned into a
maximize-convention tensor by flipping the sign of minimization
objectives.

Device / dtype
--------------
Defaults to ``cpu`` + ``double``. MPS can be selected via
``MOBOConfig.device='mps'``; the code sets dtype to float32 in that
case because MPS does not currently support float64 for many ops.

Robustness
----------
If GP fitting fails or the initial dataset is too small, the runner
falls back to a random Sobol sample so the pipeline keeps making
progress. These fallbacks are logged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lightsail.optimization.evaluator import ObjectiveEvaluator
from lightsail.optimization.initial_sampling import initial_samples
from lightsail.optimization.search_space import SearchSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & records
# ---------------------------------------------------------------------------


@dataclass
class MOBOConfig:
    """Hyper-parameters of the MOBO runner."""

    n_init: int = 16
    n_iterations: int = 20
    batch_size: int = 1
    seed: int = 42
    sampling_method: str = "sobol"   # "sobol" or "lhs"
    acqf_num_restarts: int = 5
    acqf_raw_samples: int = 64
    acqf_mc_samples: int = 128
    device: str = "cpu"              # "cpu" or "mps"
    dtype: str = "double"            # "double" or "float"
    ref_point_margin: float = 0.1    # additive margin below worst seen


@dataclass
class TrialRecord:
    """One design evaluation, whether from init or BO."""

    trial_id: int
    iteration: int
    source: str                      # "init" or "bo"
    params: np.ndarray
    params_normalized: np.ndarray
    objective_values: dict          # {name: float}
    objective_metadata: dict        # {name: {meta...}}
    feasible: bool
    constraint_penalty: float
    constraint_violations: list
    timestamp: float
    eval_time_seconds: float

    def to_dict(self) -> dict:
        return {
            "trial_id": int(self.trial_id),
            "iteration": int(self.iteration),
            "source": self.source,
            "params": np.asarray(self.params).tolist(),
            "params_normalized": np.asarray(self.params_normalized).tolist(),
            "objective_values": {k: float(v) for k, v in self.objective_values.items()},
            "objective_metadata": {
                k: {mk: (float(mv) if isinstance(mv, (int, float)) else mv)
                    for mk, mv in md.items()}
                for k, md in self.objective_metadata.items()
            },
            "feasible": bool(self.feasible),
            "constraint_penalty": float(self.constraint_penalty),
            "constraint_violations": [str(v) for v in self.constraint_violations],
            "timestamp": float(self.timestamp),
            "eval_time_seconds": float(self.eval_time_seconds),
        }


@dataclass
class RunResult:
    """Full result of a MOBO run."""

    trials: list
    pareto_indices: list
    objective_names: list
    objective_targets: dict
    config: MOBOConfig
    search_space_names: list
    search_space_bounds: list

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def pareto_trials(self) -> list:
        return [self.trials[i] for i in self.pareto_indices]

    def best_by(self, name: str) -> TrialRecord:
        """Return the trial with the best value for one objective."""
        target = self.objective_targets.get(name, "maximize")
        sign = 1.0 if target == "maximize" else -1.0
        return max(
            self.trials,
            key=lambda t: sign * float(t.objective_values.get(name, -np.inf)),
        )


# ---------------------------------------------------------------------------
# BoTorch import gate
# ---------------------------------------------------------------------------


def _import_botorch():
    """Import BoTorch + GPyTorch at call time with a clear error message."""
    import torch  # noqa: F401
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.optimize import optimize_acqf
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from gpytorch.mlls import ExactMarginalLogLikelihood

    try:  # prefer the log-space numerically stable variant
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement as _AcqFn,
        )
        acqf_name = "qLogNEHVI"
    except ImportError:  # pragma: no cover
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement as _AcqFn,
        )
        acqf_name = "qNEHVI"

    return {
        "torch": __import__("torch"),
        "SingleTaskGP": SingleTaskGP,
        "Normalize": Normalize,
        "Standardize": Standardize,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "optimize_acqf": optimize_acqf,
        "is_non_dominated": is_non_dominated,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "AcqFn": _AcqFn,
        "acqf_name": acqf_name,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MOBORunner:
    """Multi-objective Bayesian optimization loop using BoTorch."""

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        search_space: SearchSpace,
        config: MOBOConfig,
    ):
        self.evaluator = evaluator
        self.search_space = search_space
        self.config = config

        # Lazy torch import — deferred so import errors give clear messages.
        self._bt = _import_botorch()
        torch = self._bt["torch"]

        self.device = torch.device(config.device)
        if config.dtype == "double" and config.device != "cpu":
            # MPS doesn't support double for most ops — silently downgrade.
            logger.info(
                "Using float32 on device=%s (double unsupported)", config.device,
            )
            self.dtype = torch.float32
        else:
            self.dtype = torch.double if config.dtype == "double" else torch.float32

        # Objective bookkeeping — ordered list + sign map.
        self.objective_names: list = list(evaluator.objective_names)
        self.objective_targets: dict = dict(evaluator.objective_targets)
        self._signs = np.array(
            [1.0 if self.objective_targets[n] == "maximize" else -1.0
             for n in self.objective_names],
            dtype=np.float64,
        )

        self.trials: list = []
        self._trial_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        logger.info(
            "MOBO start: dim=%d, objectives=%s, init=%d, iters=%d, batch=%d, device=%s",
            self.search_space.n_dims,
            self.objective_names,
            self.config.n_init,
            self.config.n_iterations,
            self.config.batch_size,
            self.device,
        )

        self._run_initial_phase()
        for it in range(self.config.n_iterations):
            self._run_bo_iteration(it)

        return self._build_result()

    # ------------------------------------------------------------------
    # Phase 1: initial sampling
    # ------------------------------------------------------------------

    def _run_initial_phase(self) -> None:
        d = self.search_space.n_dims
        X_norm = initial_samples(
            n=self.config.n_init,
            d=d,
            method=self.config.sampling_method,
            seed=self.config.seed,
        )
        for i, row in enumerate(X_norm):
            params = self.search_space.denormalize(row)
            record = self._evaluate_and_record(
                params_normalized=row,
                params=params,
                iteration=i,
                source="init",
            )
            logger.info(
                "  init %02d/%02d  %s  penalty=%.3f",
                i + 1,
                self.config.n_init,
                {k: round(v, 3) for k, v in record.objective_values.items()},
                record.constraint_penalty,
            )

    # ------------------------------------------------------------------
    # Phase 2: BO iterations
    # ------------------------------------------------------------------

    def _run_bo_iteration(self, it_idx: int) -> None:
        torch = self._bt["torch"]

        X = self._collect_X_normalized()
        Y = self._collect_Y_maximize()

        if X.shape[0] < 2:
            logger.warning("Only %d trials so far, falling back to Sobol", X.shape[0])
            self._fallback_sample(it_idx)
            return

        X_t = torch.tensor(X, device=self.device, dtype=self.dtype)
        Y_t = torch.tensor(Y, device=self.device, dtype=self.dtype)
        bounds_t = torch.stack(
            [
                torch.zeros(self.search_space.n_dims, device=self.device, dtype=self.dtype),
                torch.ones(self.search_space.n_dims, device=self.device, dtype=self.dtype),
            ]
        )

        try:
            model = self._fit_model(X_t, Y_t)
        except Exception as e:  # pragma: no cover - numerical fallback
            logger.warning("GP fit failed (%s); falling back to Sobol", e)
            self._fallback_sample(it_idx)
            return

        # Reference point: worst seen minus a margin (in maximize convention).
        ref_point = (Y_t.min(dim=0).values - self.config.ref_point_margin).detach()

        try:
            candidates = self._suggest(model, X_t, ref_point, bounds_t)
        except Exception as e:  # pragma: no cover - numerical fallback
            logger.warning("Acquisition optimization failed (%s); Sobol fallback", e)
            self._fallback_sample(it_idx)
            return

        for row_t in candidates:
            row_np = np.clip(row_t.detach().cpu().numpy(), 0.0, 1.0)
            params = self.search_space.denormalize(row_np)
            record = self._evaluate_and_record(
                params_normalized=row_np,
                params=params,
                iteration=self.config.n_init + it_idx,
                source="bo",
            )
            logger.info(
                "  bo  %02d/%02d  %s  penalty=%.3f",
                it_idx + 1,
                self.config.n_iterations,
                {k: round(v, 3) for k, v in record.objective_values.items()},
                record.constraint_penalty,
            )

    # ------------------------------------------------------------------
    # BoTorch primitives
    # ------------------------------------------------------------------

    def _fit_model(self, X_t, Y_t):
        SingleTaskGP = self._bt["SingleTaskGP"]
        Normalize = self._bt["Normalize"]
        Standardize = self._bt["Standardize"]
        ExactMarginalLogLikelihood = self._bt["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = self._bt["fit_gpytorch_mll"]

        model = SingleTaskGP(
            train_X=X_t,
            train_Y=Y_t,
            input_transform=Normalize(d=X_t.shape[-1]),
            outcome_transform=Standardize(m=Y_t.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _suggest(self, model, X_t, ref_point, bounds_t):
        AcqFn = self._bt["AcqFn"]
        optimize_acqf = self._bt["optimize_acqf"]

        acqf = AcqFn(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=X_t,
            prune_baseline=True,
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_t,
            q=self.config.batch_size,
            num_restarts=self.config.acqf_num_restarts,
            raw_samples=self.config.acqf_raw_samples,
        )
        return candidates

    def _fallback_sample(self, it_idx: int) -> None:
        row = initial_samples(
            n=1,
            d=self.search_space.n_dims,
            method="sobol",
            seed=self.config.seed + 10_000 + it_idx,
        )[0]
        params = self.search_space.denormalize(row)
        self._evaluate_and_record(
            params_normalized=row,
            params=params,
            iteration=self.config.n_init + it_idx,
            source="bo_fallback",
        )

    # ------------------------------------------------------------------
    # Evaluation & bookkeeping
    # ------------------------------------------------------------------

    def _evaluate_and_record(
        self,
        params_normalized: np.ndarray,
        params: np.ndarray,
        iteration: int,
        source: str,
    ) -> TrialRecord:
        t0 = time.time()
        eval_result = self.evaluator.evaluate(params)
        dt = time.time() - t0

        values = {name: float(v.value) for name, v in eval_result.objective_values.items()}
        metadata = {
            name: dict(v.metadata) for name, v in eval_result.objective_values.items()
        }

        record = TrialRecord(
            trial_id=self._trial_counter,
            iteration=iteration,
            source=source,
            params=np.asarray(params, dtype=np.float64).copy(),
            params_normalized=np.asarray(params_normalized, dtype=np.float64).copy(),
            objective_values=values,
            objective_metadata=metadata,
            feasible=bool(eval_result.feasible),
            constraint_penalty=float(eval_result.constraint_penalty),
            constraint_violations=list(eval_result.constraint_violations),
            timestamp=time.time(),
            eval_time_seconds=dt,
        )
        self.trials.append(record)
        self._trial_counter += 1
        return record

    def _collect_X_normalized(self) -> np.ndarray:
        if not self.trials:
            return np.zeros((0, self.search_space.n_dims))
        return np.stack([t.params_normalized for t in self.trials])

    def _collect_Y_maximize(self) -> np.ndarray:
        """Return Y in the maximize convention (signs flipped for minimize)."""
        if not self.trials:
            return np.zeros((0, len(self.objective_names)))
        rows = []
        for trial in self.trials:
            row = np.array(
                [
                    self._signs[i] * float(trial.objective_values.get(name, 0.0))
                    for i, name in enumerate(self.objective_names)
                ],
                dtype=np.float64,
            )
            rows.append(row)
        return np.stack(rows)

    def _build_result(self) -> RunResult:
        torch = self._bt["torch"]
        is_non_dominated = self._bt["is_non_dominated"]

        Y = self._collect_Y_maximize()
        if Y.shape[0] > 0:
            Y_t = torch.tensor(Y, dtype=self.dtype)
            mask = is_non_dominated(Y_t).cpu().numpy()
            pareto_indices = np.flatnonzero(mask).tolist()
        else:
            pareto_indices = []

        return RunResult(
            trials=list(self.trials),
            pareto_indices=pareto_indices,
            objective_names=list(self.objective_names),
            objective_targets=dict(self.objective_targets),
            config=self.config,
            search_space_names=list(self.search_space.names),
            search_space_bounds=list(self.search_space.bounds),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_run_result(run: RunResult, output_dir: Path) -> None:
    """Save a :class:`RunResult` to disk.

    Produces:

    - ``trials.json``: full per-trial records + metadata
    - ``params.npy``: (n_trials, d) physical parameter matrix
    - ``objectives.npy``: (n_trials, m) objective value matrix
    - ``pareto_indices.npy``: row indices of Pareto-optimal trials
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "trials.json", "w") as f:
        json.dump(
            {
                "objective_names": run.objective_names,
                "objective_targets": run.objective_targets,
                "search_space_names": run.search_space_names,
                "search_space_bounds": [list(b) for b in run.search_space_bounds],
                "pareto_indices": list(run.pareto_indices),
                "config": asdict(run.config),
                "n_trials": run.n_trials,
                "trials": [t.to_dict() for t in run.trials],
            },
            f,
            indent=2,
            default=str,
        )

    if run.trials:
        params_arr = np.stack([t.params for t in run.trials])
        np.save(output_dir / "params.npy", params_arr)

        obj_matrix = np.array(
            [
                [t.objective_values[n] for n in run.objective_names]
                for t in run.trials
            ],
            dtype=np.float64,
        )
        np.save(output_dir / "objectives.npy", obj_matrix)
        np.save(output_dir / "pareto_indices.npy", np.array(run.pareto_indices))

    logger.info("Run saved to %s", output_dir)
