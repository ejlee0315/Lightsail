"""Fabrication constraint checking for SiN lightsail geometries.

Checks performed on a :class:`Structure`:

1. Thickness bounds
2. Minimum feature width (holes and rings)
3. Minimum gap (hole-to-hole and ring-to-ring)
4. Fill-fraction sanity range (for PhC patches)
5. Basic disconnected / impossible-geometry checks:
   - adjacent holes whose bounding-boxes would overlap
     (wall collapse)
   - ring with non-positive width
   - ring_n's inner radius smaller than ring_{n-1}'s outer radius

The checker operates in two modes:

- ``HARD``: first violation marks the design infeasible and the
  returned penalty is the sum of all violations (still computed so
  the optimizer can rank infeasible solutions among themselves).
- ``PENALTY``: always returns ``feasible=True`` and a scalar penalty
  proportional to the aggregated normalized violation.

The penalty is a dimensionless sum. Each individual violation
contributes ``(required - actual) / required`` (clipped below at
zero). That means one "fully violated" constraint contributes ~1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from lightsail.geometry.base import ParametricGeometry, Structure


class ConstraintMode(Enum):
    """Behaviour when a constraint is violated."""

    HARD = "hard"
    PENALTY = "penalty"


@dataclass
class ConstraintResult:
    """Result of evaluating a Structure against FabConstraints."""

    feasible: bool
    violations: list[str] = field(default_factory=list)
    penalty: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.feasible


@dataclass
class FabConstraints:
    """Mode-A fabrication constraints for SiN lightsail designs."""

    min_feature_nm: float = 500.0
    min_gap_nm: float = 500.0
    thickness_range_nm: tuple[float, float] = (200.0, 1000.0)
    fill_fraction_range: tuple[float, float] = (0.05, 0.60)
    fab_mode: str = "mode_a"
    mode: ConstraintMode = ConstraintMode.PENALTY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, structure: Structure) -> ConstraintResult:
        """Check all fabrication constraints and return a ConstraintResult."""
        violations: list[str] = []
        penalty = 0.0
        metrics: dict[str, float] = {}

        # 1. Thickness ----------------------------------------------------
        penalty += self._check_thickness(structure, violations, metrics)

        # 2. Hole feature / gap / disconnection --------------------------
        if structure.has_phc:
            penalty += self._check_phc(structure, violations, metrics)

        # 3. Ring feature / gap / ordering -------------------------------
        if structure.has_metagrating:
            penalty += self._check_rings(structure, violations, metrics)

        # 4. Fill-fraction sanity (only meaningful for PhC patches) ------
        if structure.has_phc:
            penalty += self._check_fill_fraction(structure, violations, metrics)

        feasible = len(violations) == 0
        if self.mode == ConstraintMode.PENALTY:
            # Always feasible in penalty mode — optimizer sees a soft cost.
            return ConstraintResult(
                feasible=True,
                violations=violations,
                penalty=float(penalty),
                metrics=metrics,
            )
        return ConstraintResult(
            feasible=feasible,
            violations=violations,
            penalty=float(penalty),
            metrics=metrics,
        )

    def clip_params(
        self,
        params: np.ndarray,
        geometry: ParametricGeometry,
    ) -> np.ndarray:
        """Clip a parameter vector to the geometry's bounds (not physical feasibility)."""
        bounds = geometry.param_bounds()
        clipped = np.copy(np.asarray(params, dtype=float))
        for i, (lo, hi) in enumerate(bounds):
            clipped[i] = np.clip(clipped[i], lo, hi)
        return clipped

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_thickness(
        self,
        structure: Structure,
        violations: list[str],
        metrics: dict[str, float],
    ) -> float:
        t = structure.thickness_nm
        t_min, t_max = self.thickness_range_nm
        metrics["thickness_nm"] = t

        if t < t_min:
            violations.append(f"thickness {t:.0f} nm < min {t_min:.0f} nm")
            return (t_min - t) / t_min
        if t > t_max:
            violations.append(f"thickness {t:.0f} nm > max {t_max:.0f} nm")
            return (t - t_max) / t_max
        return 0.0

    def _check_phc(
        self,
        structure: Structure,
        violations: list[str],
        metrics: dict[str, float],
    ) -> float:
        penalty = 0.0

        # Hole shape is shared across all sites in a PhCReflector, so
        # we inspect the first hole only for feature metrics.
        shape = structure.holes[0].shape
        min_feat = shape.min_feature_nm()
        max_ext = shape.max_extent_nm()
        bbox_w, bbox_h = shape.bounding_box_nm()
        metrics["hole_min_feature_nm"] = min_feat
        metrics["hole_max_extent_nm"] = max_ext
        metrics["hole_bbox_w_nm"] = bbox_w
        metrics["hole_bbox_h_nm"] = bbox_h

        if min_feat < self.min_feature_nm:
            violations.append(
                f"hole min feature {min_feat:.0f} nm < {self.min_feature_nm:.0f} nm"
            )
            penalty += (self.min_feature_nm - min_feat) / self.min_feature_nm

        # Gap: use lattice nearest-neighbor distance and the hole's max
        # extent as a conservative proxy for wall thickness.
        nn = structure.metadata.get("nearest_neighbor_nm")
        if nn is None and structure.lattice_period_nm is not None:
            nn = structure.lattice_period_nm
        if nn is not None:
            gap = nn - max_ext
            metrics["hole_gap_nm"] = gap
            if gap < self.min_gap_nm:
                violations.append(
                    f"hole-to-hole gap {gap:.0f} nm < {self.min_gap_nm:.0f} nm"
                )
                penalty += (self.min_gap_nm - gap) / self.min_gap_nm

            # Disconnection / impossible geometry: hole bounding box
            # larger than NN spacing means adjacent holes overlap and
            # walls collapse.
            if max(bbox_w, bbox_h) >= nn:
                violations.append(
                    f"disconnected: hole extent {max(bbox_w, bbox_h):.0f} nm "
                    f">= NN spacing {nn:.0f} nm (walls collapse)"
                )
                penalty += (max(bbox_w, bbox_h) - nn) / nn + 1.0

        return penalty

    def _check_rings(
        self,
        structure: Structure,
        violations: list[str],
        metrics: dict[str, float],
    ) -> float:
        penalty = 0.0
        rings = structure.rings

        widths = [r.width_nm for r in rings]
        if widths:
            metrics["ring_min_width_nm"] = float(min(widths))

        for i, r in enumerate(rings):
            w = r.width_nm
            if w <= 0:
                violations.append(f"ring {i} has non-positive width {w:.0f} nm")
                penalty += 1.0
                continue
            if w < self.min_feature_nm:
                violations.append(
                    f"ring {i} width {w:.0f} nm < min feature {self.min_feature_nm:.0f} nm"
                )
                penalty += (self.min_feature_nm - w) / self.min_feature_nm

        # Ring-to-ring gaps and ordering
        gaps = []
        for i in range(len(rings) - 1):
            gap = rings[i + 1].inner_radius_nm - rings[i].outer_radius_nm
            gaps.append(gap)
            if gap < 0:
                violations.append(
                    f"ring {i+1} inner radius < ring {i} outer radius (overlap)"
                )
                penalty += 1.0
                continue
            if gap < self.min_gap_nm:
                violations.append(
                    f"ring {i}-{i+1} gap {gap:.0f} nm < min gap {self.min_gap_nm:.0f} nm"
                )
                penalty += (self.min_gap_nm - gap) / self.min_gap_nm
        if gaps:
            metrics["ring_min_gap_nm"] = float(min(gaps))

        # Curvature/asymmetry sanity: if |curvature|+|asymmetry| exceeds
        # (gap/period), warped boundaries overlap into neighbors.
        if rings and structure.metadata.get("grating_period_nm"):
            period = structure.metadata["grating_period_nm"]
            max_warp = abs(rings[0].curvature) + abs(rings[0].asymmetry)
            warp_budget = max(
                0.0,
                (rings[0].width_nm * 0.5 + self.min_gap_nm * 0.5) / max(period, 1.0),
            )
            metrics["ring_warp_amplitude"] = max_warp
            if max_warp > 0.5:
                violations.append(
                    f"ring warp amplitude {max_warp:.2f} > 0.5 (likely self-overlap)"
                )
                penalty += max_warp - 0.5

        return penalty

    def _check_fill_fraction(
        self,
        structure: Structure,
        violations: list[str],
        metrics: dict[str, float],
    ) -> float:
        # Fill-fraction over the unit cell: hole area / cell area.
        if not structure.holes:
            return 0.0
        hole_area = structure.holes[0].shape.area_nm2()

        # Prefer the exact unit-cell area if it was recorded.
        cell_area = structure.metadata.get("unit_cell_area_nm2")
        if cell_area is None and structure.lattice_period_nm is not None:
            # Fall back on a square-cell approximation.
            cell_area = structure.lattice_period_nm ** 2
        if cell_area is None or cell_area <= 0:
            return 0.0

        ff = hole_area / cell_area
        metrics["fill_fraction"] = ff
        lo, hi = self.fill_fraction_range
        if ff < lo:
            violations.append(
                f"fill fraction {ff:.3f} below sanity range [{lo:.3f}, {hi:.3f}]"
            )
            return (lo - ff) / lo
        if ff > hi:
            violations.append(
                f"fill fraction {ff:.3f} above sanity range [{lo:.3f}, {hi:.3f}]"
            )
            return (ff - hi) / hi
        return 0.0
