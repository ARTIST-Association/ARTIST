from unittest import mock

import h5py
import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.scenario.configuration_classes import SurfaceConfig
from artist.util import config_dictionary


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
                config_dictionary.kinematics_type: config_dictionary.rigid_body_key,
                config_dictionary.kinematics_initial_orientation: torch.rand(4, 4),
                config_dictionary.translation_deviations: torch.rand(4, 9),
                config_dictionary.rotation_deviations: torch.rand(4, 4),
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
) -> None:
    """
    Test the heliostat field loaded from HDF5 method.

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

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_h5_file = mock.MagicMock(spec=h5py.File)
    mock_group_heliostats = mock.MagicMock()
    mock_group_heliostats.keys.return_value = ["heliostat_1"]
    mock_h5_file.__getitem__.return_value = mock_group_heliostats
    mock_group_heliostats.__len__.return_value = 1
    mock_group_heliostats.keys.return_value = ["heliostats"]

    with pytest.raises(ValueError) as exc_info:
        HeliostatField.from_hdf5(
            config_file=mock_h5_file,
            prototype_surface=prototype_surface,
            prototype_kinematics=prototype_kinematics,
            prototype_actuators=prototype_actuators,
            number_of_surface_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )
    assert error in str(exc_info.value)
