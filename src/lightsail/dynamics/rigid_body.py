"""6-DOF rigid body EoM for the spinning lightsail (Gieseler 2024 style).

Step A3 of paper-grade trajectory simulation.

State vector (12 components, MKS units)::

    s = [x, y, z, vx, vy, vz, θx, θy, θz, ωx, ωy, ωz]

where (x, y, z) is sail center-of-mass position in lab frame,
(θx, θy, θz) are small tilt angles (linearized around upright),
and (ωx, ωy, ωz) are body-fixed angular velocities. The sail is a
THIN DISC of mass M and radius R_sail; its inertia tensor is

    I_xx = I_yy = (1/4) M R²       (out-of-plane bending axes)
    I_zz = (1/2) M R²              (spin axis)

Equations of motion (Newton + Euler, lab frame for translation,
body frame for rotation, with the linearized small-angle
approximation for body→lab transforms)::

    M · d²x/dt² = F_x_lab          (and similarly for y, z)

    I_xx · dωx/dt = τ_x − (I_zz − I_yy) · ω_y · ω_z
    I_yy · dωy/dt = τ_y − (I_xx − I_zz) · ω_z · ω_x
    I_zz · dωz/dt = τ_z

    dθx/dt = ωx,   dθy/dt = ωy,   dθz/dt = ωz   (small-angle approx)

Spin around z is added as the initial condition ω_z(0) = Ω_spin.
For an axisymmetric sail (I_xx = I_yy), the gyroscopic cross term
on ω_y becomes (I_xx − I_zz) · Ω_spin · ωx — this is the
*classical gyroscopic stiffening* that resists tilt change.

The optical force/torque is queried from the polar integrator
(A2) at every RHS evaluation; for performance the LUTs are passed
through a pre-built closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from lightsail.dynamics.force_lut import CenterPhCLUT, RingLUT
from lightsail.dynamics.optical_integrator import (
    GaussianBeam,
    IntegrationConfig,
    SailGeometry,
    total_optical_force_torque,
)


C_LIGHT = 299_792_458.0


@dataclass
class SailMass:
    """Mass and inertia tensor of the rigid sail (thin uniform disc)."""

    mass_kg: float
    radius_m: float

    @property
    def I_xx(self) -> float:
        return 0.25 * self.mass_kg * self.radius_m ** 2

    @property
    def I_yy(self) -> float:
        return self.I_xx

    @property
    def I_zz(self) -> float:
        return 0.5 * self.mass_kg * self.radius_m ** 2


@dataclass
class TrajectoryResult:
    """Output of one trajectory simulation run."""

    t_s: np.ndarray
    state: np.ndarray              # shape (12, n_t)
    geometry: SailGeometry
    beam: GaussianBeam
    mass: SailMass
    spin_init_Hz: float
    initial_state: np.ndarray
    success: bool                  # ode integrator success
    message: str

    @property
    def x(self) -> np.ndarray:
        return self.state[0]

    @property
    def y(self) -> np.ndarray:
        return self.state[1]

    @property
    def z(self) -> np.ndarray:
        return self.state[2]

    @property
    def theta_x_deg(self) -> np.ndarray:
        return np.rad2deg(self.state[6])

    @property
    def theta_y_deg(self) -> np.ndarray:
        return np.rad2deg(self.state[7])

    @property
    def theta_z_deg(self) -> np.ndarray:
        return np.rad2deg(self.state[8])

    def lateral_displacement_m(self) -> np.ndarray:
        return np.hypot(self.x, self.y)

    def tilt_magnitude_deg(self) -> np.ndarray:
        return np.hypot(self.theta_x_deg, self.theta_y_deg)


def make_force_torque_callable(
    geometry: SailGeometry,
    beam: GaussianBeam,
    center_lut: CenterPhCLUT,
    ring_lut,                                    # RingLUT or RingLUT2D
    config: Optional[IntegrationConfig] = None,
    mod_amp: float = 0.0,
    n_petals: int = 0,
    base_duty: float = 0.5,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return f(t, state) → (F_x, F_y, F_z, τ_x, τ_y, τ_z) Lab frame, for the EoM.

    With ``mod_amp > 0`` and ``n_petals > 0`` (and a ``RingLUT2D``), the
    integrator uses azimuthal-modulated ring with sail-frame angle
    φ_sail = φ_lab − θ_z(t), where θ_z is the current spin angle.
    """
    cfg = config or IntegrationConfig()

    def force_torque(t: float, state: np.ndarray) -> np.ndarray:
        x, y, _z = state[0], state[1], state[2]
        theta_x, theta_y, theta_z = state[6], state[7], state[8]
        out = total_optical_force_torque(
            sail_x_m=float(x),
            sail_y_m=float(y),
            sail_theta_x_rad=float(theta_x),
            sail_theta_y_rad=float(theta_y),
            geometry=geometry,
            beam=beam,
            center_lut=center_lut,
            ring_lut=ring_lut,
            config=cfg,
            sail_yaw_rad=float(theta_z),
            mod_amp=mod_amp,
            n_petals=n_petals,
            base_duty=base_duty,
        )
        return np.array([
            out.F_x_N, out.F_y_N, out.F_z_N,
            out.tau_x_Nm, out.tau_y_Nm, out.tau_z_Nm,
        ], dtype=float)

    return force_torque


def build_rhs(
    mass: SailMass,
    force_torque_fn: Callable[[float, np.ndarray], np.ndarray],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return scipy-compatible RHS function for the 12-D state ODE."""
    Ix, Iy, Iz = mass.I_xx, mass.I_yy, mass.I_zz
    M = mass.mass_kg

    def rhs(t: float, s: np.ndarray) -> np.ndarray:
        F_x, F_y, F_z, tau_x, tau_y, tau_z = force_torque_fn(t, s)
        x, y, z, vx, vy, vz, thx, thy, thz, wx, wy, wz = s

        # Translational
        ax = F_x / M
        ay = F_y / M
        az = F_z / M
        # Rotational (Euler equations with cross terms)
        dwx = (tau_x - (Iz - Iy) * wy * wz) / Ix
        dwy = (tau_y - (Ix - Iz) * wz * wx) / Iy
        dwz = tau_z / Iz
        # Small-angle approx: dθ/dt ≈ ω
        return np.array([
            vx, vy, vz,
            ax, ay, az,
            wx, wy, wz,
            dwx, dwy, dwz,
        ], dtype=float)

    return rhs


def run_trajectory(
    initial_position_m: tuple[float, float, float],
    initial_tilt_rad: tuple[float, float, float],
    spin_freq_Hz: float,
    geometry: SailGeometry,
    beam: GaussianBeam,
    mass: SailMass,
    center_lut: CenterPhCLUT,
    ring_lut,                                # RingLUT or RingLUT2D
    integration_config: Optional[IntegrationConfig] = None,
    t_end_s: float = 5.0,
    n_eval: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    mod_amp: float = 0.0,
    n_petals: int = 0,
    base_duty: float = 0.5,
) -> TrajectoryResult:
    """Integrate 6-DOF rigid body EoM over [0, t_end] with given perturbation."""
    s0 = np.zeros(12, dtype=float)
    s0[0:3] = initial_position_m
    s0[6:9] = initial_tilt_rad
    s0[11] = 2.0 * np.pi * float(spin_freq_Hz)   # ω_z initial spin

    force_torque_fn = make_force_torque_callable(
        geometry, beam, center_lut, ring_lut, integration_config,
        mod_amp=mod_amp, n_petals=n_petals, base_duty=base_duty,
    )
    rhs = build_rhs(mass, force_torque_fn)
    t_eval = np.linspace(0.0, t_end_s, n_eval)

    sol = solve_ivp(
        rhs, (0.0, t_end_s), s0, t_eval=t_eval,
        method="RK45", rtol=rtol, atol=atol,
    )
    return TrajectoryResult(
        t_s=sol.t,
        state=sol.y,
        geometry=geometry,
        beam=beam,
        mass=mass,
        spin_init_Hz=spin_freq_Hz,
        initial_state=s0,
        success=sol.success,
        message=str(sol.message),
    )
