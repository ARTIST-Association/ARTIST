import pathlib
from unittest import mock

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.field import HeliostatField
from artist.io import h5_columnar_heliostats
from artist.scenario import Scenario
from artist.util import constants
from artist.util.config import SurfaceConfig
from tests.field.heliostat_field_state import assert_states_equal, heliostat_group_state


@pytest.mark.parametrize(
    "prototype_surface, prototype_kinematics, prototype_actuators, error",
    [
        (
            None,
            None,
            None,
            "If the heliostat does not have individual surface parameters, a surface prototype must be provided!",
        ),
        (
            mock.MagicMock(),
            None,
            None,
            "If the heliostat does not have an individual kinematics, a kinematics prototype must be provided!",
        ),
        (
            mock.MagicMock(),
            {
                constants.kinematics_type: constants.rigid_body_key,
                constants.kinematics_initial_orientation: torch.rand(4, 4),
                constants.translation_deviations: torch.rand(4, 9),
                constants.rotation_deviations: torch.rand(4, 4),
            },
            None,
            "If the heliostat does not have individual actuators, an actuator prototype must be provided!",
        ),
    ],
)
def test_heliostat_field_load_from_hdf5_errors(
    prototype_surface: SurfaceConfig,
    prototype_kinematics: dict[str, str | torch.Tensor],
    prototype_actuators: dict[str, str | torch.Tensor],
    error: str,
    device: torch.device,
    tmp_path: pathlib.Path,
) -> None:
    """
    Test that loading heliostats without individual parameters requires the prototypes.

    Parameters
    ----------
    prototype_surface : SurfaceConfig
        The mock prototype surface.
    prototype_kinematics : dict[str, str | torch.Tensor]
        The mock prototype kinematics.
    prototype_actuators : dict[str, str | torch.Tensor]
        The mock prototype actuator.
    error : str
        The expected error message.
    device : torch.device
        The device on which to initialize tensors.
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    with h5py.File(tmp_path / "scenario.h5", "w") as scenario_file:
        h5_columnar_heliostats.write_columnar_heliostats(
            scenario_file.create_group(constants.heliostat_key),
            h5_columnar_heliostats.heliostat_columns(
                [
                    (
                        "heliostat_1",
                        {
                            constants.heliostat_id: 1,
                            constants.heliostat_position: torch.tensor(
                                [0.0, 0.0, 0.0, 1.0]
                            ),
                        },
                    )
                ]
            ),
        )

    with (
        h5py.File(tmp_path / "scenario.h5", "r") as scenario_file,
        pytest.raises(ValueError) as exc_info,
    ):
        HeliostatField.from_hdf5(
            config_file=scenario_file,
            prototype_surface=prototype_surface,
            prototype_kinematics=prototype_kinematics,
            prototype_actuators=prototype_actuators,
            number_of_surface_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )
    assert error in str(exc_info.value)


@pytest.mark.parametrize(
    "scenario_name",
    [
        "test_scenario_paint_four_heliostats",
        "test_scenario_paint_mix_ideal_prototype_deflectometry",
    ],
)
def test_surface_evaluation_chunk_size_does_not_change_result(
    scenario_name: str,
) -> None:
    """
    Test that evaluating the surfaces in smaller chunks loads exactly the same heliostat field.

    Parameters
    ----------
    scenario_name : str
        The name of the test scenario.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_path = (
        pathlib.Path(ARTIST_ROOT) / f"tests/data/scenarios/{scenario_name}.h5"
    )
    states_per_chunk_size = []
    for surface_evaluation_chunk_size in (1024, 1, 3):
        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=torch.tensor([10, 10]),
                device=torch.device("cpu"),
                surface_evaluation_chunk_size=surface_evaluation_chunk_size,
            )
        states_per_chunk_size.append(
            [
                heliostat_group_state(heliostat_group)
                for heliostat_group in scenario.heliostat_field.heliostat_groups
            ]
        )

    for states in states_per_chunk_size[1:]:
        assert_states_equal(states_per_chunk_size[0], states)


@pytest.mark.parametrize("surface_evaluation_chunk_size", [0, -1])
def test_surface_evaluation_chunk_size_must_be_positive(
    surface_evaluation_chunk_size: int,
) -> None:
    """
    Test that a non-positive surface evaluation chunk size is rejected.

    Parameters
    ----------
    surface_evaluation_chunk_size : int
        The invalid chunk size.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    with pytest.raises(ValueError) as exc_info:
        HeliostatField.from_hdf5(
            config_file=mock.MagicMock(),
            prototype_surface=mock.MagicMock(),
            prototype_kinematics={},
            prototype_actuators={},
            number_of_surface_points_per_facet=torch.tensor([10, 10]),
            device=torch.device("cpu"),
            surface_evaluation_chunk_size=surface_evaluation_chunk_size,
        )
    assert "The surface evaluation chunk size must be positive" in str(exc_info.value)
