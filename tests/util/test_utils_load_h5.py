import logging

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.util import config_dictionary, utils_load_h5


@pytest.mark.parametrize(
    "kinematic_type",
    [(
        "invalid_kinematic_type"
    )],
)
def test_load_kinematic_deviations(
        mocker: MockerFixture,
        kinematic_type: str, 
        device: torch.device
    ):
    """
    Test errors raised when loading kinematic deviations from an hdf5 file.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    kinematic_type : str
        The kinematic type to be tested.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_file = mocker.MagicMock(spec=h5py.File)

    mock_level_1 = mocker.MagicMock()

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_kinematic_key: mock_level_1
    }[key]

    log = mocker.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        utils_load_h5.kinematic_deviations(
            prototype=False,
            kinematic_type=kinematic_type,
            scenario_file=scenario_file,
            log=log,
            device=device,
        )

    assert f"The kinematic type: {kinematic_type} is not yet implemented!" in str(
        exc_info.value
    )
