"""Rotor assembly in blade modal coordinates.

Blade modes are an input. Nothing here forms a blade stiffness or solves a
blade eigenproblem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class BladeModalModel:
    """One blade, as the plain arrays a single-blade modal analysis produces.

    Parameters
    ----------
    frequencies_hz : array_like, shape (n_modes,)
        Undamped natural frequencies in hertz.
    mode_shapes : array_like, shape (n_dof, n_modes)
        Column ``j`` is mode ``j``. This is the ``n_dof`` by ``n_modes``
        layout 1d-fem writes for one blade.
    mass : array_like, shape (n_modes,) or (n_dof, n_dof)
        Generalized modal masses ``phi_j^T M phi_j``, or the constrained mass
        matrix in the same degrees of freedom as ``mode_shapes``. Modes are
        taken as mass-orthogonal, so only those diagonal terms are used.
    """

    frequencies_hz: np.ndarray
    mode_shapes: np.ndarray
    mass: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64).reshape(-1)
        shapes = np.asarray(self.mode_shapes, dtype=np.float64)
        mass = np.asarray(self.mass, dtype=np.float64)
        if shapes.ndim != 2:
            raise ValueError("mode_shapes must have shape (n_dof, n_modes)")
        n_dof, n_modes = shapes.shape
        if n_modes == 0 or n_dof == 0:
            raise ValueError("mode_shapes must contain at least one degree of freedom and one mode")
        if frequencies.shape != (n_modes,):
            raise ValueError("frequencies_hz length must equal the number of mode-shape columns")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("frequencies_hz must be finite and positive")
        if mass.shape == (n_modes,):
            pass
        elif mass.shape == (n_dof, n_dof):
            if not np.allclose(mass, mass.T, rtol=0.0, atol=1e-12 * max(1.0, np.max(np.abs(mass)))):
                raise ValueError("mass matrix must be symmetric")
        else:
            raise ValueError("mass must have shape (n_modes,) or (n_dof, n_dof)")
        if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(mass)):
            raise ValueError("mode_shapes and mass must be finite")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "mode_shapes", shapes)
        object.__setattr__(self, "mass", mass)
        if np.any(generalized_modal_mass(self) <= 0.0):
            raise ValueError("generalized modal masses phi_j^T M phi_j must be positive")


@dataclass(frozen=True)
class HubShaft:
    """Hub and shaft that carry the blades.

    Parameters
    ----------
    hub_mass_kg : float
        Hub mass.
    hub_inertia_kg_m2 : float
        Hub polar inertia about the shaft axis.
    shaft_stiffness_nm_per_rad : float
        Shaft torsional stiffness. Non-negative.
    azimuth_rad : array_like, shape (n_blades,), optional
        Blade azimuths. Omitted when the caller does not place the blades.
    """

    hub_mass_kg: float
    hub_inertia_kg_m2: float
    shaft_stiffness_nm_per_rad: float
    azimuth_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        hub_mass = float(self.hub_mass_kg)
        hub_inertia = float(self.hub_inertia_kg_m2)
        shaft_stiffness = float(self.shaft_stiffness_nm_per_rad)
        for name, value in (
            ("hub_mass_kg", hub_mass),
            ("hub_inertia_kg_m2", hub_inertia),
            ("shaft_stiffness_nm_per_rad", shaft_stiffness),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        azimuth = None
        if self.azimuth_rad is not None:
            azimuth = np.asarray(self.azimuth_rad, dtype=np.float64).reshape(-1)
            if azimuth.size == 0 or not np.all(np.isfinite(azimuth)):
                raise ValueError("azimuth_rad must be a non-empty finite array")
        object.__setattr__(self, "hub_mass_kg", hub_mass)
        object.__setattr__(self, "hub_inertia_kg_m2", hub_inertia)
        object.__setattr__(self, "shaft_stiffness_nm_per_rad", shaft_stiffness)
        object.__setattr__(self, "azimuth_rad", azimuth)


@dataclass(frozen=True)
class AerodynamicLoadSeries:
    """Aerodynamic loads of the kind a blade-element tool returns.

    Parameters
    ----------
    time_s : array_like, shape (n_steps,)
        Sample times. Strictly increasing when there is more than one sample.
    loads : array_like, shape (n_steps, n_blades, n_dof)
        Nodal forces in the same degree-of-freedom order as each blade's
        mode shapes. Every blade in the series shares ``n_dof``.
    """

    time_s: np.ndarray
    loads: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        loads = np.asarray(self.loads, dtype=np.float64)
        if time.size == 0:
            raise ValueError("time_s must contain at least one sample")
        if loads.ndim != 3:
            raise ValueError("loads must have shape (n_steps, n_blades, n_dof)")
        if loads.shape[0] != time.shape[0]:
            raise ValueError("loads and time_s must have the same number of samples")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(loads)):
            raise ValueError("time_s and loads must be finite")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "loads", loads)


@dataclass(frozen=True)
class RotorResult:
    """Assembled rotor frequencies and, when loads were supplied, the modal response.

    Parameters
    ----------
    frequencies_hz : ndarray, shape (n_rotor_modes,)
        Undamped assembled frequencies in hertz, ascending. ``n_rotor_modes``
        is the sum of the blade mode counts.
    modal_response : ndarray, shape (n_steps, n_blades, n_modes), optional
        Modal coordinates ``q`` of each blade. Physical displacement of blade
        ``b`` at a sample is ``mode_shapes @ q``. ``None`` when no loads were
        supplied.
    rotor_speed_rad_s : float
        Rotor speed about the shaft axis, in radians per second.
    """

    frequencies_hz: np.ndarray
    modal_response: np.ndarray | None
    rotor_speed_rad_s: float


def generalized_modal_mass(blade: BladeModalModel) -> np.ndarray:
    """Return ``phi_j^T M phi_j`` for each mode.

    Parameters
    ----------
    blade : BladeModalModel
        Blade modal model. ``mass`` is either those scalars or the matrix ``M``.

    Returns
    -------
    ndarray, shape (n_modes,)
    """
    mass = blade.mass
    n_modes = blade.mode_shapes.shape[1]
    if mass.shape == (n_modes,):
        return mass.copy()
    shapes = blade.mode_shapes
    return np.einsum("ij,ik,kj->j", shapes, mass, shapes)


def assemble_rotor(
    blades: list[BladeModalModel] | tuple[BladeModalModel, ...],
    hub_shaft: HubShaft,
    rotor_speed_rad_s: float,
    loads: AerodynamicLoadSeries | None = None,
) -> RotorResult:
    """Assemble blade modal models on a hub and shaft.

    Each supplied mode is one rotor degree of freedom. Root-fixed cantilever
    shapes do not move the hub, so the hub mass, hub inertia and shaft
    stiffness do not couple the blades. The rotor equations in modal
    coordinates are

        M q_ddot + Omega G q_dot + K q = Q

    with ``M`` and ``K`` diagonal, ``K_jj = M_jj omega_j^2``, and ``G = 0``.
    ``Omega`` is ``rotor_speed_rad_s``. The undamped frequencies are therefore
    the supplied blade frequencies, repeated once for each blade, at every
    rotor speed.

    Parameters
    ----------
    blades : sequence of BladeModalModel
        One entry per blade.
    hub_shaft : HubShaft
        Hub and shaft. ``azimuth_rad``, when given, has one entry per blade.
    rotor_speed_rad_s : float
        Rotor speed about the shaft axis. Finite. The sign is the sense of rotation.
    loads : AerodynamicLoadSeries, optional
        BEM force series. When given, every blade must have the same number of
        modes and the same number of degrees of freedom.

    Returns
    -------
    RotorResult
        Ascending assembled frequencies, and the modal response when ``loads``
        is given.
    """
    if len(blades) == 0:
        raise ValueError("at least one blade is required")
    if any(not isinstance(blade, BladeModalModel) for blade in blades):
        raise TypeError("blades must be BladeModalModel instances")
    if not isinstance(hub_shaft, HubShaft):
        raise TypeError("hub_shaft must be a HubShaft")
    speed = float(rotor_speed_rad_s)
    if not np.isfinite(speed):
        raise ValueError("rotor_speed_rad_s must be finite")
    if hub_shaft.azimuth_rad is not None and hub_shaft.azimuth_rad.shape != (len(blades),):
        raise ValueError("azimuth_rad length must equal the number of blades")

    modal_mass = np.concatenate([generalized_modal_mass(blade) for blade in blades])
    omega = _TWO_PI * np.concatenate([blade.frequencies_hz for blade in blades])
    stiffness = modal_mass * omega**2
    # G is zero for these root-fixed shapes, so Omega G drops out of
    # lambda^2 M + lambda Omega G + K = 0 and lambda^2 = -K/M.
    gyroscopic = np.zeros_like(modal_mass)
    omega_squared = stiffness / modal_mass - (speed * gyroscopic) ** 2 / (4.0 * modal_mass**2)
    frequencies = np.sqrt(omega_squared) / _TWO_PI
    frequencies = np.sort(frequencies, kind="mergesort")

    response = None if loads is None else _modal_response(blades, loads)
    return RotorResult(
        frequencies_hz=frequencies,
        modal_response=response,
        rotor_speed_rad_s=speed,
    )


def _modal_response(blades: tuple[BladeModalModel, ...] | list[BladeModalModel], loads: AerodynamicLoadSeries) -> np.ndarray:
    """Undamped modal response, constant-average-acceleration Newmark, zero initial state."""
    if not isinstance(loads, AerodynamicLoadSeries):
        raise TypeError("loads must be an AerodynamicLoadSeries")
    n_blades = len(blades)
    n_steps = loads.time_s.shape[0]
    n_modes = blades[0].mode_shapes.shape[1]
    n_dof = blades[0].mode_shapes.shape[0]
    if loads.loads.shape[1] != n_blades:
        raise ValueError("loads n_blades must equal the number of blades")
    if loads.loads.shape[2] != n_dof:
        raise ValueError("loads n_dof must match the blade mode shapes")
    for blade in blades:
        if blade.mode_shapes.shape != (n_dof, n_modes):
            raise ValueError("a load series requires every blade to share n_dof and n_modes")

    response = np.zeros((n_steps, n_blades, n_modes), dtype=np.float64)
    for index, blade in enumerate(blades):
        mass = generalized_modal_mass(blade)
        stiffness = mass * (_TWO_PI * blade.frequencies_hz) ** 2
        modal_force = loads.loads[:, index, :] @ blade.mode_shapes
        response[:, index, :] = _newmark_modal(loads.time_s, modal_force, mass, stiffness)
    return response


def _newmark_modal(
    time_s: np.ndarray,
    modal_force: np.ndarray,
    modal_mass: np.ndarray,
    modal_stiffness: np.ndarray,
) -> np.ndarray:
    """Integrate ``m q_ddot + k q = Q(t)`` with Newmark beta = 1/4, gamma = 1/2."""
    n_steps = time_s.shape[0]
    coordinates = np.zeros_like(modal_force)
    if n_steps == 1:
        coordinates[0] = modal_force[0] / modal_stiffness
        return coordinates

    beta = 0.25
    gamma = 0.5
    velocity = np.zeros(modal_mass.shape[0], dtype=np.float64)
    acceleration = (modal_force[0] - modal_stiffness * coordinates[0]) / modal_mass
    for step in range(n_steps - 1):
        dt = float(time_s[step + 1] - time_s[step])
        inv_beta_dt2 = 1.0 / (beta * dt * dt)
        inv_beta_dt = 1.0 / (beta * dt)
        history = 1.0 / (2.0 * beta) - 1.0
        effective_stiffness = modal_mass * inv_beta_dt2 + modal_stiffness
        effective_force = modal_force[step + 1] + modal_mass * (
            inv_beta_dt2 * coordinates[step] + inv_beta_dt * velocity + history * acceleration
        )
        next_coordinates = effective_force / effective_stiffness
        next_acceleration = (
            inv_beta_dt2 * (next_coordinates - coordinates[step]) - inv_beta_dt * velocity - history * acceleration
        )
        next_velocity = velocity + dt * ((1.0 - gamma) * acceleration + gamma * next_acceleration)
        coordinates[step + 1] = next_coordinates
        velocity = next_velocity
        acceleration = next_acceleration
    return coordinates
