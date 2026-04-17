from lightsail.simulation.base import ElectromagneticSolver
from lightsail.simulation.mock import MockSolver
from lightsail.simulation.results import SimulationResult

try:
    from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver

    __all__ = [
        "ElectromagneticSolver",
        "MockSolver",
        "RCWASolver",
        "RCWAConfig",
        "SimulationResult",
    ]
except ImportError:  # grcwa optional
    __all__ = ["ElectromagneticSolver", "MockSolver", "SimulationResult"]
