"""Polar geometric integration of optical force/torque over the sail.

Step A2 of paper-grade trajectory simulation. Given the sail state
(position + tilt), the geometry (center disc + outer ring annulus),
the Gaussian laser beam profile, and the per-area force LUTs (A1),
compute the total optical force and torque on the rigid sail.

Geometry conventions (sail frame, with sail normal along +ẑ_sail)::

    Center disc:    0      ≤ r ≤ R_inner    (PhC reflector zone)
    Outer ring:    R_inner ≤ r ≤ R_outer    (concentric metagrating)

Lab frame: laser propagates in +ẑ direction. Sail at position
(sail_x, sail_y) in transverse plane, tilted by (θ_x, θ_y) around
the lab x, y axes. Spin around z (yaw, θ_z) does not affect
optics for an axisymmetric ring (concentric, asym = curv = 0).

Local incidence angle: at a sail point (r, φ in sail frame), the
photon's in-plane wavevector in sail frame is

    k_in_xy_sail ≈ k0 · (sin θ_y, -sin θ_x)         (small-angle approx)

The radial outward unit vector at that point is
``(cos φ, sin φ, 0)``. Project k_in_xy onto the radial direction
to get the local effective incidence angle for the local 1D ring
grating::

    sin θ_local = sin θ_y · cos φ − sin θ_x · sin φ

For the central PhC (axisymmetric circular hole pattern) the local
angle is just the magnitude of sail tilt at that point.

Local laser intensity (Gaussian beam centered at lab origin)::

    I(x_lab, y_lab) = I₀ · exp(−2 (x²+y²) / w²)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lightsail.dynamics.force_lut import CenterPhCLUT, RingLUT


C_LIGHT = 299_792_458.0


@dataclass
class SailGeometry:
    """Two-zone sail geometry in metres.

    ``curvature_radius_m`` controls the sail's parabolic shape::

        z(r) = − r² / (2 · R_c)         convex toward the laser if R_c > 0
        z(r) = 0                         flat sail (R_c = +∞)
        z(r) = + r² / (2 · R_c)          concave toward the laser if R_c < 0
                                         (negative R_c value = concave)

    The convention is that R_c > 0 (convex) is the Salary 2019 stable
    geometry: each off-center surface element's outward-normal tilts
    toward the local radial direction by angle r / R_c, which (combined
    with off-axis Gaussian intensity gradient) produces a lateral
    restoring force when the sail is displaced.

    The local incidence angle correction is added by the polar
    integrator (see ``total_optical_force_torque``).
    """

    R_inner_m: float       # center PhC outer radius
    R_outer_m: float       # outer ring outer radius (= sail edge)
    curvature_radius_m: float = float("inf")     # +∞ = flat sail

    @property
    def center_area_m2(self) -> float:
        return float(np.pi * self.R_inner_m ** 2)

    @property
    def ring_area_m2(self) -> float:
        return float(np.pi * (self.R_outer_m ** 2 - self.R_inner_m ** 2))

    @property
    def total_area_m2(self) -> float:
        return float(np.pi * self.R_outer_m ** 2)

    def curvature_tilt(self, r: float) -> float:
        """Local outward tilt of the sail normal at radius r [rad].

        Positive value = local normal tilts OUTWARD (radially) for a
        convex-toward-laser sail (R_c > 0). For a flat sail (R_c = ∞)
        this returns 0.
        """
        if not np.isfinite(self.curvature_radius_m):
            return 0.0
        # Small-slope approximation: tan α = r / |R_c|, sign = sign(R_c).
        return float(r / self.curvature_radius_m)


@dataclass
class GaussianBeam:
    """Gaussian beam profile centered at lab origin (x_lab=y_lab=0)."""

    I0_W_per_m2: float
    waist_m: float
    wavelength_nm: float = 1550.0

    def intensity(self, x_lab_m: np.ndarray, y_lab_m: np.ndarray) -> np.ndarray:
        d2 = x_lab_m ** 2 + y_lab_m ** 2
        return self.I0_W_per_m2 * np.exp(-2.0 * d2 / self.waist_m ** 2)


@dataclass
class OpticalForceTorque:
    """Result of one geometric integration."""

    F_x_N: float
    F_y_N: float
    F_z_N: float
    tau_x_Nm: float
    tau_y_Nm: float
    tau_z_Nm: float
    F_z_center_N: float
    F_z_ring_N: float
    F_radial_ring_max_per_area_Pa: float


@dataclass
class IntegrationConfig:
    """Numerical knobs for polar (r, φ) integration."""

    n_radial_center: int = 8        # radial samples in center disc
    n_radial_ring: int = 12         # radial samples in ring annulus
    n_azimuthal: int = 64           # azimuthal samples
    use_wavelength_lookup: bool = False  # False = use band-mean LUT


def total_optical_force_torque(
    sail_x_m: float,
    sail_y_m: float,
    sail_theta_x_rad: float,
    sail_theta_y_rad: float,
    geometry: SailGeometry,
    beam: GaussianBeam,
    center_lut: CenterPhCLUT,
    ring_lut: RingLUT,
    config: Optional[IntegrationConfig] = None,
) -> OpticalForceTorque:
    """Polar (r, φ) integration of force/torque on a tilted sail in a Gaussian beam.

    Sail spin (yaw) is not modelled here — for axisymmetric concentric ring
    the optics are spin-invariant. Spin's gyroscopic contribution is added
    at the EoM level (A3), not here.
    """
    cfg = config or IntegrationConfig()

    # Build (r, φ) grid for center and ring separately so we can use
    # different radial densities.
    r_center = np.linspace(0.0, geometry.R_inner_m, cfg.n_radial_center, endpoint=False)
    r_center = r_center + (r_center[1] - r_center[0] if cfg.n_radial_center > 1 else 0) / 2.0
    dr_center = (geometry.R_inner_m / cfg.n_radial_center)

    r_ring = np.linspace(
        geometry.R_inner_m, geometry.R_outer_m, cfg.n_radial_ring, endpoint=False
    )
    r_ring = r_ring + (r_ring[1] - r_ring[0] if cfg.n_radial_ring > 1 else 0) / 2.0
    dr_ring = (geometry.R_outer_m - geometry.R_inner_m) / cfg.n_radial_ring

    phi = np.linspace(0.0, 2.0 * np.pi, cfg.n_azimuthal, endpoint=False)
    dphi = 2.0 * np.pi / cfg.n_azimuthal

    sin_tx = float(np.sin(sail_theta_x_rad))
    sin_ty = float(np.sin(sail_theta_y_rad))

    F_x_total = 0.0
    F_y_total = 0.0
    F_z_total = 0.0
    F_z_center_total = 0.0
    F_z_ring_total = 0.0
    tau_x_total = 0.0
    tau_y_total = 0.0
    tau_z_total = 0.0
    Fr_max = 0.0

    cos_phi_all = np.cos(phi)
    sin_phi_all = np.sin(phi)

    # Curvature → outward tilt α(r) = r / R_c at each radius r (sign = sign(R_c)).
    # Convex toward laser (R_c > 0) → positive α.
    # The local sin θ_radial at point (r, φ) becomes
    #   sin θ_local_radial = sin θ_y · cos φ − sin θ_x · sin φ + α(r)
    # (the curvature contribution is uniform in φ at fixed r — radially symmetric).

    # ------- Center PhC disc (vectorized over φ) -------
    # Center is axisymmetric; the LUT was sweeped only in θ_in along
    # one direction, so we use its band-mean response. We take the
    # *magnitude* of the local effective tilt direction including
    # both sail tilt and curvature.
    for r in r_center:
        alpha_r = geometry.curvature_tilt(r)
        # Local radial direction at this point varies with φ; the
        # effective tilt magnitude at each (r, φ):
        sin_th_radial = sin_ty * cos_phi_all - sin_tx * sin_phi_all + alpha_r
        # Center PhC LUT depends on tilt magnitude (axisymmetric pattern):
        sin_th_mag = np.sqrt(sin_th_radial ** 2)   # = |sin_th_radial| for axisymmetric
        theta_local_deg = np.rad2deg(np.arcsin(np.clip(sin_th_mag, 0.0, 1.0)))
        Fz_per_area_phi, _ = center_lut.force_per_area_vec(theta_local_deg, 1.0)

        x_lab = sail_x_m + r * cos_phi_all
        y_lab = sail_y_m + r * sin_phi_all
        I_local = beam.intensity(x_lab, y_lab)
        dA = r * dr_center * dphi
        F_z_pts = Fz_per_area_phi * I_local * dA
        s = float(np.sum(F_z_pts))
        F_z_total += s
        F_z_center_total += s
        # Torque from axial F_z alone: τ_x = +r sin φ · F_z, τ_y = -r cos φ · F_z
        tau_x_total += float(np.sum((r * sin_phi_all) * F_z_pts))
        tau_y_total += float(np.sum(-(r * cos_phi_all) * F_z_pts))

    # ------- Outer ring annulus (vectorized over φ; per-r LUT call) -------
    for r in r_ring:
        alpha_r = geometry.curvature_tilt(r)
        sin_th_local_arr = sin_ty * cos_phi_all - sin_tx * sin_phi_all + alpha_r
        theta_local_deg_arr = np.rad2deg(np.arcsin(np.clip(sin_th_local_arr, -1.0, 1.0)))
        Fz_per_area_phi, Fr_in_per_area_phi = ring_lut.force_per_area_vec(
            theta_local_deg_arr, 1.0,
        )

        x_lab = sail_x_m + r * cos_phi_all
        y_lab = sail_y_m + r * sin_phi_all
        I_local = beam.intensity(x_lab, y_lab)

        dA_phi = r * dr_ring * dphi
        F_z_pts = Fz_per_area_phi * I_local * dA_phi
        Fr_in_pts = Fr_in_per_area_phi * I_local * dA_phi
        Fr_max = max(Fr_max, float(np.max(np.abs(Fr_in_per_area_phi * I_local))))

        # Ring force at point φ in sail frame:
        #   F_radial_inward_vec = - F_r_inward · (cos φ, sin φ)
        #   F_z_vec = + F_z · ẑ
        F_x_pts = -Fr_in_pts * cos_phi_all
        F_y_pts = -Fr_in_pts * sin_phi_all
        F_x_total += float(np.sum(F_x_pts))
        F_y_total += float(np.sum(F_y_pts))
        F_z_total += float(np.sum(F_z_pts))
        F_z_ring_total += float(np.sum(F_z_pts))

        # Torque about sail-frame origin:  τ = r × F  (in 3D)
        #   r_pt = (r cos φ, r sin φ, 0)   F_pt = (F_x_pts, F_y_pts, F_z_pts)
        #   (r×F)_x =  r_y · F_z =  r sin φ · F_z
        #   (r×F)_y = -r_x · F_z = -r cos φ · F_z
        #   (r×F)_z =  r_x · F_y − r_y · F_x = r cos φ · F_y − r sin φ · F_x
        tau_x_total += float(np.sum((r * sin_phi_all) * F_z_pts))
        tau_y_total += float(np.sum(-(r * cos_phi_all) * F_z_pts))
        tau_z_total += float(np.sum((r * cos_phi_all) * F_y_pts - (r * sin_phi_all) * F_x_pts))

    return OpticalForceTorque(
        F_x_N=F_x_total,
        F_y_N=F_y_total,
        F_z_N=F_z_total,
        tau_x_Nm=tau_x_total,
        tau_y_Nm=tau_y_total,
        tau_z_Nm=tau_z_total,
        F_z_center_N=F_z_center_total,
        F_z_ring_N=F_z_ring_total,
        F_radial_ring_max_per_area_Pa=Fr_max,
    )
