"""Lightsail dynamics — paper-grade trajectory and stability analysis.

Implements the Gieseler 2024 (Nat Commun 15:4203) approach adapted to
our 2-zone architecture (central PhC reflector + outer concentric
ring metagrating)::

    * force_lut       — per-area force/torque lookup tables from RCWA/FMM
    * optical_integrator — polar (r, φ) integration over sail with Gaussian beam
    * rigid_body      — 6-DOF rigid body EoM with spin (scipy.solve_ivp)
    * floquet         — monodromy matrix + eigenvalue analysis
    * trajectory      — full simulation with paper-style PASS/FAIL verdict
"""
from __future__ import annotations

try:
    from lightsail.dynamics.force_lut import (
        CenterPhCLUT,
        RingLUT,
        compute_center_lut,
        compute_ring_lut,
    )
    from lightsail.dynamics.optical_integrator import (
        GaussianBeam,
        IntegrationConfig,
        OpticalForceTorque,
        SailGeometry,
        total_optical_force_torque,
    )
    from lightsail.dynamics.rigid_body import (
        SailMass,
        TrajectoryResult,
        build_rhs,
        make_force_torque_callable,
        run_trajectory,
    )
    from lightsail.dynamics.floquet import (
        StabilityResult,
        classify_eigenvalues,
        compute_eigenvalues_linear,
        compute_jacobian,
        compute_monodromy,
    )

    __all__ = [
        "CenterPhCLUT",
        "RingLUT",
        "compute_center_lut",
        "compute_ring_lut",
        "SailGeometry",
        "GaussianBeam",
        "IntegrationConfig",
        "OpticalForceTorque",
        "total_optical_force_torque",
        "SailMass",
        "TrajectoryResult",
        "build_rhs",
        "make_force_torque_callable",
        "run_trajectory",
        "StabilityResult",
        "compute_jacobian",
        "compute_eigenvalues_linear",
        "compute_monodromy",
        "classify_eigenvalues",
    ]
except ImportError:  # grcwa optional
    __all__ = []
