"""Rotor assembly reports one stage and does not solve blade modes again."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from rotor_dynamics import BladeModalModel, HubShaft, assemble_rotor
from rotor_dynamics.stage_log import (
    LOGGER_NAME,
    StageCollector,
    StageReport,
    collecting,
    fold_reports,
    stage,
)


def test_public_contract() -> None:
    assert LOGGER_NAME == "blade.stage"
    assert [item.name for item in dataclasses.fields(StageReport)] == [
        "package", "stage", "seconds", "error", "depth", "started"]
    assert callable(stage) and callable(fold_reports)
    assert issubclass(StageCollector, logging.Handler)


def test_a_stage_records_seconds_and_an_error_keeps_its_time() -> None:
    with collecting() as sink:
        with stage("rotor_dynamics", "kept"):
            pass
        with pytest.raises(RuntimeError):
            with stage("rotor_dynamics", "failed"):
                raise RuntimeError("stage failed")
    folded = fold_reports(sink.take())
    assert folded["kept"] >= 0.0
    failed = folded["failed"]
    assert isinstance(failed, dict)
    assert failed["seconds"] >= 0.0
    assert failed["error"] == "stage failed"


def test_assemble_rotor_reports_the_assembly_and_keeps_blade_frequencies() -> None:
    frequencies = np.array([1.5, 4.0])
    shapes = np.array([[0.0, 0.0], [0.5, -0.2], [1.0, 1.0]])
    mass = np.ones(2)
    blade = BladeModalModel(frequencies_hz=frequencies, mode_shapes=shapes, mass=mass)
    hub = HubShaft(hub_mass_kg=10.0, hub_inertia_kg_m2=2.0, shaft_stiffness_nm_per_rad=1.0e5)
    with collecting() as sink:
        result = assemble_rotor((blade, blade), hub, rotor_speed_rad_s=0.0)
    expected = np.sort(np.concatenate((frequencies, frequencies)))
    np.testing.assert_allclose(result.frequencies_hz, expected, rtol=1e-12, atol=0.0)
    folded = fold_reports(sink.take())
    assert set(folded) == {"rotor_assembly"}
    assert folded["rotor_assembly"] >= 0.0
