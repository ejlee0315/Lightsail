"""Linearized stability + Floquet analysis (A4).

For an **axisymmetric** sail (concentric ring, no asymmetry), the
optical force is rotation-invariant about the spin (z) axis, so the
linearized EoM is **autonomous** — no time-periodic forcing exists,
and Floquet analysis reduces to ordinary linear-stability eigenvalue
analysis of the 12×12 Jacobian at the equilibrium.

For asymmetric / non-rotation-invariant designs (`asymmetry` ≠ 0 in the
MetaGrating), the optical force depends on the spin angle θ_z(t) =
Ω_spin · t, and Floquet analysis is required: integrate the
fundamental matrix of the linearized system over one spin period
T_spin = 1/Ω_spin and compute eigenvalues of the monodromy matrix.

This module exposes:
    * compute_jacobian(state_eq, dynamics)            — 12×12 numerical Jacobian
    * compute_eigenvalues_linear(...)                 — autonomous case (axisymmetric)
    * compute_monodromy(...)                          — Floquet for time-periodic case
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class StabilityResult:
    """Eigenvalue-based stability classification."""

    eigenvalues: np.ndarray   # complex
    max_real_part: float
    stable: bool              # max Re(λ) < 1e-6
    marginal: bool            # max Re(λ) within ±1e-6
    classification: str       # "stable" | "marginal" | "unstable"


def compute_jacobian(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    state_eq: np.ndarray,
    t: float = 0.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian ∂(rhs)/∂(state) at equilibrium via centered finite difference."""
    n = state_eq.size
    J = np.zeros((n, n), dtype=float)
    for k in range(n):
        sp = state_eq.copy(); sp[k] += eps
        sm = state_eq.copy(); sm[k] -= eps
        J[:, k] = (rhs(t, sp) - rhs(t, sm)) / (2.0 * eps)
    return J


def classify_eigenvalues(eigenvalues: np.ndarray, tol: float = 1e-6) -> StabilityResult:
    re = np.real(eigenvalues)
    max_re = float(re.max())
    stable = max_re < -tol
    marginal = abs(max_re) <= tol
    if stable:
        cls = "stable"
    elif marginal:
        cls = "marginal"
    else:
        cls = "unstable"
    return StabilityResult(
        eigenvalues=eigenvalues,
        max_real_part=max_re,
        stable=stable,
        marginal=marginal,
        classification=cls,
    )


def compute_eigenvalues_linear(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    state_eq: np.ndarray,
    eps: float = 1e-5,
) -> StabilityResult:
    """Autonomous linear stability: eigenvalues of Jacobian J = ∂rhs/∂state.

    ``stable`` if all Re(λ) < 0 (system decays to equilibrium).
    ``marginal`` if max Re(λ) ≈ 0 (oscillates without growing).
    ``unstable`` if max Re(λ) > 0 (perturbations grow exponentially).
    """
    J = compute_jacobian(rhs, state_eq, eps=eps)
    eigs = np.linalg.eigvals(J)
    return classify_eigenvalues(eigs)


def compute_monodromy(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    state_eq: np.ndarray,
    period_s: float,
    eps: float = 1e-5,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> StabilityResult:
    """Floquet monodromy matrix M and its eigenvalues for periodic systems.

    Integrates the variational equation dΦ/dt = J(t)·Φ from Φ(0)=I to
    Φ(T_spin)=M and returns eigenvalues. Stable if all |λ_M| ≤ 1
    (marginal at |λ_M| = 1).
    """
    n = state_eq.size
    Phi0 = np.eye(n).flatten()

    def variational_rhs(t, vec):
        Phi = vec.reshape((n, n))
        J = compute_jacobian(rhs, state_eq, t=t, eps=eps)
        dPhi = J @ Phi
        return dPhi.flatten()

    sol = solve_ivp(
        variational_rhs, (0.0, period_s), Phi0,
        method="RK45", rtol=rtol, atol=atol,
    )
    M = sol.y[:, -1].reshape((n, n))
    eigs = np.linalg.eigvals(M)
    # For Floquet, "stable" is |λ| < 1, marginal at |λ| = 1
    abs_max = float(np.max(np.abs(eigs)))
    if abs_max < 1.0 - 1e-6:
        cls = "stable"; stable = True; marginal = False
    elif abs_max <= 1.0 + 1e-6:
        cls = "marginal"; stable = False; marginal = True
    else:
        cls = "unstable"; stable = False; marginal = False
    return StabilityResult(
        eigenvalues=eigs,
        max_real_part=abs_max,
        stable=stable,
        marginal=marginal,
        classification=cls,
    )
