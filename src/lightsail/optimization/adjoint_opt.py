"""Adjoint-based gradient optimization for freeform lightsail PhC design.

Uses grcwa's autograd backend to compute dR/d(epsilon_grid) in a single
backward pass, then updates the permittivity grid via projected gradient
descent with a minimum-feature-size (MFS) morphological filter.

The design variable is a *continuous density field* ρ(x,y) ∈ [0, 1],
which is mapped to permittivity via:

    ε(x,y) = ε_air + ρ(x,y) × (ε_SiN - ε_air)

and filtered through a conic filter + threshold projection to enforce
minimum feature size.

This module is self-contained and does NOT use the BO pipeline. It can
be used standalone:

    python -m lightsail.optimization.adjoint_opt --launch 1550 --niter 200

or imported and called from a script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MFS filter: conic filter + threshold projection (standard TO approach)
# ---------------------------------------------------------------------------


def _conic_filter(rho_2d: np.ndarray, radius_px: float) -> np.ndarray:
    """Apply a conic density filter via FFT convolution (autograd-safe).

    Convolves the density field with a cone-shaped kernel of the given
    radius (in pixels). Uses FFT-based convolution so it works with
    autograd's traced arrays.
    """
    Nx, Ny = rho_2d.shape

    # Build cone kernel (plain numpy, not traced)
    r = int(np.ceil(radius_px))
    if r < 1:
        return rho_2d.flatten()
    ky, kx = np.mgrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(kx.astype(float) ** 2 + ky.astype(float) ** 2)
    kernel = np.maximum(radius_px - dist, 0.0)
    kernel = kernel / kernel.sum()

    # Pad kernel to grid size
    pad = np.zeros((Nx, Ny))
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    # Center the kernel (circular shift)
    pad = np.roll(np.roll(pad, -(kh // 2), axis=0), -(kw // 2), axis=1)

    # FFT convolution — autograd traces through anp.fft operations
    try:
        import autograd.numpy as anp
        F_rho = anp.fft.fft2(rho_2d)
        F_ker = anp.fft.fft2(anp.array(pad))
        result = anp.real(anp.fft.ifft2(F_rho * F_ker))
    except Exception:
        F_rho = np.fft.fft2(rho_2d)
        F_ker = np.fft.fft2(pad)
        result = np.real(np.fft.ifft2(F_rho * F_ker))

    return result.flatten()


def _threshold_projection(rho, beta: float, eta: float = 0.5):
    """Smooth Heaviside projection to binarize the density field.

    Higher beta → sharper threshold → more binary design.
    Works with both numpy and autograd arrays.
    """
    try:
        import autograd.numpy as anp
        num = anp.tanh(beta * eta) + anp.tanh(beta * (rho - eta))
        den = anp.tanh(beta * eta) + anp.tanh(beta * (1.0 - eta))
    except Exception:
        num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


# ---------------------------------------------------------------------------
# Autograd-compatible forward model
# ---------------------------------------------------------------------------


def _make_forward_fn(
    nG: int,
    L1_um: list,
    L2_um: list,
    thickness_um: float,
    Nx: int,
    Ny: int,
    n_sin: float,
):
    """Return a function R(rho_flat) that is differentiable via autograd.

    ``rho_flat`` is a (Nx*Ny,) array with values in [0, 1].

    Uses frequency-dependent complex permittivity from the SiN dispersion
    model for physical accuracy. The autograd path only traces the real
    part of ε (imaginary k is negligible in NIR for SiN).
    """
    from grcwa import set_backend

    set_backend("autograd")
    import autograd.numpy as anp
    import grcwa

    # Precompute SiN permittivity for each wavelength we'll need.
    # The dispersion object is called outside the autograd trace.
    try:
        from lightsail.materials.sin import SiNDispersion
        _disp = SiNDispersion()
        _use_dispersion = True
    except ImportError:
        _use_dispersion = False

    eps_air = 1.0

    def _get_eps_sin(freq):
        """Get real part of SiN permittivity at given frequency (1/um)."""
        if _use_dispersion:
            wl_nm = 1000.0 / freq  # freq in 1/um → wl in nm
            return float(complex(_disp.epsilon(wl_nm)).real)
        return n_sin**2

    def forward(rho_flat, freq):
        """Compute reflectance for a single frequency."""
        eps_sin = _get_eps_sin(freq)
        eps_grid = eps_air + rho_flat * (eps_sin - eps_air)

        # p-polarization
        sim = grcwa.obj(nG, list(L1_um), list(L2_um), freq, 0.0, 0.0,
                        verbose=0)
        sim.Add_LayerUniform(0, eps_air)
        sim.Add_LayerGrid(thickness_um, Nx, Ny)
        sim.Add_LayerUniform(0, eps_air)
        sim.Init_Setup()
        sim.GridLayer_geteps(eps_grid.reshape(Nx, Ny))
        sim.MakeExcitationPlanewave(1, 0, 0, 0)
        Rp, Tp = sim.RT_Solve(normalize=1)

        # s-polarization
        sim2 = grcwa.obj(nG, list(L1_um), list(L2_um), freq, 0.0, 0.0,
                         verbose=0)
        sim2.Add_LayerUniform(0, eps_air)
        sim2.Add_LayerGrid(thickness_um, Nx, Ny)
        sim2.Add_LayerUniform(0, eps_air)
        sim2.Init_Setup()
        sim2.GridLayer_geteps(eps_grid.reshape(Nx, Ny))
        sim2.MakeExcitationPlanewave(0, 0, 1, 0)
        Rs, Ts = sim2.RT_Solve(normalize=1)

        return (Rp + Rs) / 2.0

    return forward


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------


@dataclass
class AdjointConfig:
    """Configuration for adjoint-based topology optimization."""

    # Grid
    Nx: int = 96
    Ny: int = 96
    nG: int = 21  # lower than BO's 41 for speed; validate at 41 after

    # Structure
    thickness_nm: float = 240.0
    period_nm: float = 1580.0
    n_sin: float = 2.0  # approximate SiN index at 1550 nm

    # Optimization
    n_iterations: int = 200
    learning_rate: float = 0.02
    beta_init: float = 2.0       # initial projection sharpness
    beta_max: float = 64.0       # final projection sharpness
    beta_increase_every: int = 40  # increase beta every N iterations
    mfs_radius_nm: float = 100.0  # minimum feature size (radius)

    # Wavelength
    launch_wavelength_nm: float = 1550.0
    n_wavelengths: int = 10  # sample points in Doppler range
    beta_final: float = 0.2  # target velocity

    # Init
    init_hole_radius_frac: float = 0.39  # initial circular hole (like trial 23)
    init_mode: str = "circle"   # "circle", "noisy_circle", "random", "multi"
    init_noise_amplitude: float = 0.15   # noise amplitude for "noisy_circle"
    init_seed: int = 42
    n_random_starts: int = 5    # for "multi" mode: run N random inits, keep best

    # Output
    output_dir: Optional[str] = None


def run_adjoint_optimization(config: AdjointConfig) -> dict:
    """Run adjoint-based freeform optimization. Returns best design info."""
    from autograd import grad
    import autograd.numpy as anp

    logger.info("Adjoint optimization: %d iterations, %dx%d grid, nG=%d",
                config.n_iterations, config.Nx, config.Ny, config.nG)

    period_um = config.period_nm / 1000.0
    thickness_um = config.thickness_nm / 1000.0
    sqrt3 = np.sqrt(3.0)

    # Triangular lattice vectors
    L1_um = [period_um, 0.0]
    L2_um = [period_um * 0.5, period_um * sqrt3 * 0.5]

    # Wavelength grid (Doppler range)
    lam0 = config.launch_wavelength_nm
    lam_max = lam0 * np.sqrt((1 + config.beta_final) / (1 - config.beta_final))
    wavelengths_nm = np.linspace(lam0, lam_max, config.n_wavelengths)
    freqs = 1.0 / (wavelengths_nm / 1000.0)  # 1/um

    # MFS filter radius in pixels
    mfs_radius_px = config.mfs_radius_nm / config.period_nm * config.Nx

    # Build forward function
    forward = _make_forward_fn(
        config.nG, L1_um, L2_um, thickness_um,
        config.Nx, config.Ny, config.n_sin,
    )

    # Initialize density field
    rng = np.random.RandomState(config.init_seed)
    x = np.linspace(0, 1, config.Nx, endpoint=False)
    y = np.linspace(0, 1, config.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)

    if config.init_mode == "circle":
        rho = np.where(dist < config.init_hole_radius_frac, 0.0, 1.0).flatten()
    elif config.init_mode == "noisy_circle":
        # Circular hole + random noise → breaks symmetry
        base = np.where(dist < config.init_hole_radius_frac, 0.2, 0.8).flatten()
        noise = rng.randn(config.Nx * config.Ny) * config.init_noise_amplitude
        rho = np.clip(base + noise, 0.0, 1.0)
    elif config.init_mode == "random":
        # Fully random → explores diverse topologies
        rho = rng.rand(config.Nx * config.Ny) * 0.6 + 0.2  # [0.2, 0.8]
    else:
        rho = np.where(dist < config.init_hole_radius_frac, 0.0, 1.0).flatten()

    # Smooth initial design with MFS filter
    rho = _conic_filter(rho.reshape(config.Nx, config.Ny), mfs_radius_px)
    rho = np.clip(rho, 0.01, 0.99)
    logger.info("Init mode: %s, initial fill=%.3f", config.init_mode, np.mean(rho > 0.5))

    beta_proj = config.beta_init
    best_R = -np.inf
    best_rho = rho.copy()
    history = []

    logger.info("Wavelengths: %.0f - %.0f nm (%d points)",
                wavelengths_nm[0], wavelengths_nm[-1], len(wavelengths_nm))
    logger.info("MFS filter radius: %.1f px (%.0f nm)", mfs_radius_px, config.mfs_radius_nm)

    for it in range(config.n_iterations):
        # Apply filter + projection
        rho_filtered = _conic_filter(
            rho.reshape(config.Nx, config.Ny), mfs_radius_px
        )
        rho_proj = _threshold_projection(rho_filtered, beta_proj)
        rho_proj = np.clip(rho_proj, 0.01, 0.99)

        # Compute mean R across Doppler wavelengths + gradient
        total_R = 0.0
        total_grad = np.zeros_like(rho)

        for freq in freqs:
            def make_obj(f):
                def objective(rho_flat):
                    rho_2d = rho_flat.reshape(config.Nx, config.Ny)
                    rho_f = _conic_filter(rho_2d, mfs_radius_px)
                    rho_p = _threshold_projection(rho_f, beta_proj)
                    rho_p = anp.clip(rho_p, 0.01, 0.99)
                    return forward(rho_p, f)
                return objective

            objective = make_obj(freq)
            R_val = float(forward(rho_proj, freq))
            total_R += R_val

            dR = grad(objective)
            g = dR(rho)
            total_grad += np.array(g)

        mean_R = total_R / len(freqs)
        total_grad /= len(freqs)

        # Gradient ascent (maximize R)
        rho = rho + config.learning_rate * total_grad
        rho = np.clip(rho, 0.0, 1.0)

        # Beta continuation
        if (it + 1) % config.beta_increase_every == 0 and beta_proj < config.beta_max:
            beta_proj = min(beta_proj * 2, config.beta_max)
            logger.info("  beta_proj → %.1f", beta_proj)

        # Track best
        if mean_R > best_R:
            best_R = mean_R
            best_rho = rho_proj.copy()

        history.append({"iteration": it, "mean_R": mean_R, "beta_proj": beta_proj})

        if it % 10 == 0 or it == config.n_iterations - 1:
            fill = 1.0 - np.mean(rho_proj < 0.5)
            logger.info(
                "  iter %3d/%d  mean_R=%.4f  best=%.4f  fill=%.2f  beta=%.1f",
                it, config.n_iterations, mean_R, best_R, fill, beta_proj,
            )

    # Final binarized design
    rho_final = _conic_filter(
        best_rho.reshape(config.Nx, config.Ny), mfs_radius_px
    )
    rho_final = _threshold_projection(rho_final, config.beta_max)
    rho_binary = (rho_final > 0.5).astype(float)

    result = {
        "best_mean_R": float(best_R),
        "rho_continuous": best_rho.reshape(config.Nx, config.Ny),
        "rho_binary": rho_binary.reshape(config.Nx, config.Ny),
        "history": history,
        "config": config,
        "wavelengths_nm": wavelengths_nm,
    }

    # Save if output dir specified
    if config.output_dir:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "rho_binary.npy", rho_binary)
        np.save(out / "rho_continuous.npy", best_rho)
        np.save(out / "history.npy", np.array([(h["iteration"], h["mean_R"]) for h in history]))
        logger.info("Saved to %s", out)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_multi_start(config: AdjointConfig) -> dict:
    """Run adjoint optimization from multiple random initial designs.

    Tries ``config.n_random_starts`` random initializations plus one
    noisy-circle start, keeps the best result. This overcomes the
    symmetry-trapping problem of starting from a pure circle.
    """
    best_result = None
    all_results = []

    inits = [("noisy_circle", config.init_seed)]
    for i in range(config.n_random_starts):
        inits.append(("random", config.init_seed + 100 + i))

    for init_mode, seed in inits:
        cfg = AdjointConfig(
            Nx=config.Nx, Ny=config.Ny, nG=config.nG,
            thickness_nm=config.thickness_nm, period_nm=config.period_nm,
            n_sin=config.n_sin,
            n_iterations=config.n_iterations,
            learning_rate=config.learning_rate,
            beta_init=config.beta_init, beta_max=config.beta_max,
            beta_increase_every=config.beta_increase_every,
            mfs_radius_nm=config.mfs_radius_nm,
            launch_wavelength_nm=config.launch_wavelength_nm,
            n_wavelengths=config.n_wavelengths,
            beta_final=config.beta_final,
            init_mode=init_mode,
            init_seed=seed,
            init_noise_amplitude=config.init_noise_amplitude,
        )
        logger.info("=== Multi-start: mode=%s seed=%d ===", init_mode, seed)
        result = run_adjoint_optimization(cfg)
        all_results.append((init_mode, seed, result["best_mean_R"]))
        logger.info("  → best R = %.4f", result["best_mean_R"])

        if best_result is None or result["best_mean_R"] > best_result["best_mean_R"]:
            best_result = result

    logger.info("=== Multi-start summary ===")
    for mode, seed, R in all_results:
        tag = " ← BEST" if R == best_result["best_mean_R"] else ""
        logger.info("  %s (seed=%d): R=%.4f%s", mode, seed, R, tag)

    return best_result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Adjoint freeform PhC optimization")
    parser.add_argument("--launch", type=float, default=1550.0)
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--nG", type=int, default=41)
    parser.add_argument("--Nx", type=int, default=96)
    parser.add_argument("--thickness", type=float, default=240.0)
    parser.add_argument("--period", type=float, default=1580.0)
    parser.add_argument("--lr", type=float, default=0.10)
    parser.add_argument("--mfs", type=float, default=80.0, help="MFS radius in nm")
    parser.add_argument("--n-wl", type=int, default=8)
    parser.add_argument("--init", type=str, default="multi",
                        choices=["circle", "noisy_circle", "random", "multi"])
    parser.add_argument("--n-starts", type=int, default=5,
                        help="Number of random starts for multi mode")
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = f"results/{timestamp}_adjoint_{args.init}_{args.launch:.0f}nm"

    config = AdjointConfig(
        Nx=args.Nx, Ny=args.Nx,
        nG=args.nG,
        thickness_nm=args.thickness,
        period_nm=args.period,
        n_iterations=args.niter,
        learning_rate=args.lr,
        mfs_radius_nm=args.mfs,
        launch_wavelength_nm=args.launch,
        n_wavelengths=args.n_wl,
        output_dir=out_dir,
        init_mode=args.init,
        init_noise_amplitude=args.noise,
        init_seed=args.seed,
        n_random_starts=args.n_starts,
    )

    import time
    t0 = time.time()

    if args.init == "multi":
        result = run_multi_start(config)
    else:
        result = run_adjoint_optimization(config)

    elapsed = time.time() - t0

    # Save
    if config.output_dir:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "rho_binary.npy", result["rho_binary"])
        np.save(out / "rho_continuous.npy", result["rho_continuous"])
        logger.info("Saved to %s", out)

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Best mean R (Doppler range): {result['best_mean_R']:.4f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
