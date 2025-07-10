import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.data_loader import stral_loader


@pytest.mark.parametrize(
    "stral_file_path, expected",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/test_stral_data.binp",
            [
                torch.tensor(
                    [
                        [-0.807500004768, 0.642499983311, 0.040198374540],
                        [0.807500004768, 0.642499983311, 0.040198374540],
                        [-0.807500004768, -0.642499983311, 0.040198374540],
                        [0.807500004768, -0.642499983311, 0.040198374540],
                    ]
                ),
                torch.tensor(
                    [
                        [8.024845123291e-01, 0.000000000000e00, 4.984567407519e-03],
                        [-1.956921551027e-05, 6.374922394753e-01, 3.150522708893e-03],
                    ]
                ),
                torch.tensor([0.007499999832, -1.147500038147, 0.037725076079]),
                torch.tensor([-0.007797845639, -0.002935287543, 0.999965310097]),
            ],
        ),
    ],
)
def test_extract_paint_heliostats(
    stral_file_path: pathlib.Path,
    expected: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the ``STRAL`` loader.

    Parameters
    ----------
    stral_file_path : pathlib.Path
        Name of the ``STRAL`` file.
    expected : list[torch.Tensor],
        The expected data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    extracted_list = list(
        stral_loader.extract_stral_deflectometry_data(
            stral_file_path=stral_file_path,
            device=device,
        )
    )

    torch.testing.assert_close(extracted_list[0], expected[0].to(device))
    torch.testing.assert_close(extracted_list[1][1], expected[1].to(device))
    torch.testing.assert_close(extracted_list[2][3][0], expected[2].to(device))
    torch.testing.assert_close(extracted_list[3][1][2], expected[3].to(device))
