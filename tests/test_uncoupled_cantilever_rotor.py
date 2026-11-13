"""Two identical root-fixed blades on a non-rotating hub stay uncoupled."""

import numpy as np

from rotor_dynamics import BladeModalModel, HubShaft, assemble_rotor

# Euler-Bernoulli fixed-free eigenvalues beta_n L for EI = mu = L = 1.
_BETA_L = np.array([1.875104068404204, 4.694091132974175, 7.854757438237612])


def _cantilever_shape(beta_l: float, stations: np.ndarray, length: float) -> np.ndarray:
    beta = beta_l / length
    sigma = (np.cosh(beta_l) + np.cos(beta_l)) / (np.sinh(beta_l) + np.sin(beta_l))
    bx = beta * stations
    return (np.cosh(bx) - np.cos(bx)) - sigma * (np.sinh(bx) - np.sin(bx))


def _cantilever_blade() -> BladeModalModel:
    """Three analytical cantilever bending modes, sampled on the free stations."""
    length = 1.0
    n_stations = 5
    dx = length / (n_stations - 1)
    stations = np.linspace(0.0, length, n_stations)[1:]
    shapes = np.column_stack([_cantilever_shape(beta_l, stations, length) for beta_l in _BETA_L])
    frequencies_hz = _BETA_L**2 / (2.0 * np.pi)
    mass = np.diag(np.full(stations.shape[0], dx))
    return BladeModalModel(frequencies_hz=frequencies_hz, mode_shapes=shapes, mass=mass)


def test_two_identical_nonrotating_cantilevers_keep_blade_frequencies():
    blade = _cantilever_blade()
    hub_shaft = HubShaft(
        hub_mass_kg=250.0,
        hub_inertia_kg_m2=40.0,
        shaft_stiffness_nm_per_rad=5.0e6,
        azimuth_rad=(0.0, np.pi),
    )

    result = assemble_rotor((blade, blade), hub_shaft, rotor_speed_rad_s=0.0)

    expected = np.sort(np.concatenate((blade.frequencies_hz, blade.frequencies_hz)))
    assert result.frequencies_hz.shape == expected.shape
    assert np.all(np.diff(result.frequencies_hz) >= 0.0)
    np.testing.assert_allclose(result.frequencies_hz, expected, rtol=1e-12, atol=0.0)
    assert result.modal_response is None
    assert result.rotor_speed_rad_s == 0.0
