from lightsail.optimization.evaluator import (
    EvaluationResult,
    ObjectiveEvaluator,
)
from lightsail.optimization.objectives import (
    AsymmetryStabilizationProxy,
    FabricationPenaltyObjective,
    MassAndFabPenaltyObjective,
    MIREmissivityObjective,
    NIRReflectivityObjective,
    Objective,
    ObjectiveContext,
    ObjectiveValue,
    RadialMomentumProxy,
    AccelerationTimeObjective,
    SailArealDensityObjective,
    StabilizationProxy,
    StabilizationProxyObjective,
    make_stage1_objectives,
    make_stage2_objectives,
)
from lightsail.optimization.optimizer import BayesianOptimizer
from lightsail.optimization.search_space import SearchSpace

try:
    from lightsail.optimization.fmm_proxy import LocalPeriodFMMProxy
    _HAS_FMM_PROXY = True
except ImportError:  # grcwa optional (StabilizationProxy doesn't require it)
    _HAS_FMM_PROXY = False

__all__ = [
    # search / optimizer
    "SearchSpace",
    "BayesianOptimizer",
    # objectives
    "Objective",
    "ObjectiveContext",
    "ObjectiveValue",
    "NIRReflectivityObjective",
    "MIREmissivityObjective",
    "MassAndFabPenaltyObjective",
    "FabricationPenaltyObjective",
    "AccelerationTimeObjective",
    "SailArealDensityObjective",
    "StabilizationProxy",
    "AsymmetryStabilizationProxy",
    "RadialMomentumProxy",
    "StabilizationProxyObjective",
    *( ("LocalPeriodFMMProxy",) if _HAS_FMM_PROXY else () ),
    "make_stage1_objectives",
    "make_stage2_objectives",
    # evaluator
    "ObjectiveEvaluator",
    "EvaluationResult",
]
