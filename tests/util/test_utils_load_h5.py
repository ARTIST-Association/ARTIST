import logging
from unittest.mock import MagicMock

import h5py
import pytest
import torch
from pytest_mock import mocker

from artist.util import config_dictionary, utils_load_h5


@pytest.mark.parametrize(
    "kinematic_type, expected",
    [()],
)
def test_load_kinematic_deviations(kinematic_type: str, device: torch.device):
    scenario_file = MagicMock(spec=h5py.File)

    mock_level_1 = mocker.MagicMock()

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_kinematic_key: mock_level_1
    }[key]

    log = MagicMock(spec=logging.Logger)

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
