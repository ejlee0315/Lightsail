from lightsail.simulation.base import ElectromagneticSolver
from lightsail.simulation.mock import MockSolver
from lightsail.simulation.results import SimulationResult

try:
    from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver
    from lightsail.simulation.layered_rcwa import LayeredRCWASolver, LayerSpec
    from lightsail.simulation.grating_fmm import (
        FMMGratingConfig,
        GratingOrderResult,
        aggregate_metagrating_response,
        compute_dC_pr_dtheta,
        compute_lateral_coefficients,
        evaluate_1d_grating,
    )
    from lightsail.simulation.stiffness import (
        StiffnessResult,
        compute_stiffness_matrix,
    )
    from lightsail.simulation.damping import (
        compute_damping_force,
        doppler_factor,
        lorentz_gamma,
        sail_frame_wavelength_nm,
    )

    __all__ = [
        "ElectromagneticSolver",
        "MockSolver",
        "RCWASolver",
        "RCWAConfig",
        "LayeredRCWASolver",
        "LayerSpec",
        "FMMGratingConfig",
        "GratingOrderResult",
        "aggregate_metagrating_response",
        "compute_dC_pr_dtheta",
        "compute_lateral_coefficients",
        "evaluate_1d_grating",
        "StiffnessResult",
        "compute_stiffness_matrix",
        "compute_damping_force",
        "doppler_factor",
        "lorentz_gamma",
        "sail_frame_wavelength_nm",
        "SimulationResult",
    ]
except ImportError:  # grcwa optional
    __all__ = ["ElectromagneticSolver", "MockSolver", "SimulationResult"]
