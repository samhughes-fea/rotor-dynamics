"""Modal response and blade-mode participation of the assembled rotor."""

import numpy as np

from rotor_dynamics import AerodynamicLoadSeries, BladeModalModel, HubShaft, assemble_rotor

_TWO_PI = 2.0 * np.pi


def _hub() -> HubShaft:
    return HubShaft(hub_mass_kg=10.0, hub_inertia_kg_m2=2.0, shaft_stiffness_nm_per_rad=1.0e5)


def _oscillator(frequency_hz: float, mass: float = 1.0) -> BladeModalModel:
    """One mass-normalised translation mode."""
    return BladeModalModel(
        frequencies_hz=np.array([frequency_hz]),
        mode_shapes=np.array([[1.0]]),
        mass=np.array([mass]),
    )


def test_one_sample_load_is_the_static_modal_coordinate() -> None:
    blade = _oscillator(1.0 / _TWO_PI)
    force = 2.0
    loads = AerodynamicLoadSeries(time_s=[0.0], loads=[[[force]]])

    result = assemble_rotor((blade,), _hub(), rotor_speed_rad_s=0.0, loads=loads)

    stiffness = 1.0 * (_TWO_PI * blade.frequencies_hz[0]) ** 2
    expected = force / stiffness
    np.testing.assert_allclose(result.modal_response, [[[expected]]], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        blade.mode_shapes @ result.modal_response[0, 0], [expected], rtol=1e-12, atol=0.0
    )


def test_unloaded_blade_stays_at_rest_when_the_other_is_forced() -> None:
    slow = _oscillator(1.0)
    fast = _oscillator(2.0)
    loads = AerodynamicLoadSeries(time_s=[0.0], loads=[[[3.0], [0.0]]])

    result = assemble_rotor((slow, fast), _hub(), rotor_speed_rad_s=0.0, loads=loads)

    assert result.modal_response.shape == (1, 2, 1)
    stiffness = 1.0 * (_TWO_PI * 1.0) ** 2
    np.testing.assert_allclose(result.modal_response[0, 0, 0], 3.0 / stiffness, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.modal_response[0, 1, 0], 0.0, rtol=0.0, atol=0.0)


def test_newmark_first_step_matches_average_acceleration() -> None:
    mass = 2.0
    frequency_hz = 0.5
    blade = _oscillator(frequency_hz, mass=mass)
    force = 4.0
    dt = 0.01
    loads = AerodynamicLoadSeries(time_s=[0.0, dt], loads=[[[force]], [[force]]])

    result = assemble_rotor((blade,), _hub(), rotor_speed_rad_s=1.3, loads=loads)

    stiffness = mass * (_TWO_PI * frequency_hz) ** 2
    acceleration = force / mass
    beta = 0.25
    effective_stiffness = mass / (beta * dt * dt) + stiffness
    history = 1.0 / (2.0 * beta) - 1.0
    effective_force = force + mass * history * acceleration
    expected = effective_force / effective_stiffness
    np.testing.assert_allclose(result.modal_response[0, 0, 0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.modal_response[1, 0, 0], expected, rtol=1e-12, atol=0.0)


def test_participation_is_one_blade_mode_per_rotor_mode() -> None:
    slow = BladeModalModel(
        frequencies_hz=np.array([1.0, 4.0]),
        mode_shapes=np.array([[1.0, 0.0], [0.0, 1.0]]),
        mass=np.ones(2),
    )
    fast = BladeModalModel(
        frequencies_hz=np.array([2.0, 3.0]),
        mode_shapes=np.array([[1.0, 0.0], [0.0, 1.0]]),
        mass=np.ones(2),
    )

    result = assemble_rotor((slow, fast), _hub(), rotor_speed_rad_s=5.0)

    np.testing.assert_allclose(result.frequencies_hz, [1.0, 2.0, 3.0, 4.0], rtol=1e-12, atol=0.0)
    # Rotor modes in frequency order: slow-0, fast-0, fast-1, slow-1.
    expected = np.zeros((4, 2, 2))
    expected[0, 0, 0] = 1.0
    expected[1, 1, 0] = 1.0
    expected[2, 1, 1] = 1.0
    expected[3, 0, 1] = 1.0
    np.testing.assert_array_equal(result.participation, expected)
    np.testing.assert_allclose(result.participation.sum(axis=(1, 2)), np.ones(4), rtol=0.0, atol=0.0)
    assert result.modal_response is None


def test_unequal_mode_counts_pad_participation() -> None:
    one = BladeModalModel(
        frequencies_hz=np.array([1.5]),
        mode_shapes=np.array([[1.0], [0.5]]),
        mass=np.array([1.0]),
    )
    two = BladeModalModel(
        frequencies_hz=np.array([0.5, 2.5]),
        mode_shapes=np.array([[1.0, 0.0], [0.0, 1.0]]),
        mass=np.ones(2),
    )

    result = assemble_rotor((one, two), _hub(), rotor_speed_rad_s=0.0)

    np.testing.assert_allclose(result.frequencies_hz, [0.5, 1.5, 2.5], rtol=1e-12, atol=0.0)
    assert result.participation.shape == (3, 2, 2)
    expected = np.zeros((3, 2, 2))
    expected[0, 1, 0] = 1.0
    expected[1, 0, 0] = 1.0
    expected[2, 1, 1] = 1.0
    np.testing.assert_array_equal(result.participation, expected)
