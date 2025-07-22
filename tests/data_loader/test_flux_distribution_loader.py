import pathlib

import torch

from artist import ARTIST_ROOT
from artist.data_loader import flux_distribution_loader


def test_load_flux_from_png(device: torch.device) -> None:
    """
    Test the function to load flux distributions from .png files.
    
    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.
    
    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    heliostat_flux_path_mapping = [
        (
            "AA39",
            [
                pathlib.Path(ARTIST_ROOT)
                /  "tests/data/field_data/AA39-flux-centered_1.png",
                pathlib.Path(ARTIST_ROOT)
                /  "tests/data/field_data/AA39-flux-centered_2.png",
            ]
        ),
        (
            "AA31",
            [
                pathlib.Path(ARTIST_ROOT)
                /  "tests/data/field_data/AA31-flux-centered_1.png",
            ]
        )
    ]
    heliostat_names = ["AA31", "AA35", "AA39", "AB38"]

    extracted_bitmaps = flux_distribution_loader.load_flux_from_png(
        heliostat_flux_path_mapping=heliostat_flux_path_mapping,
        heliostat_names=heliostat_names,
        device=device
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_fluxes_from_png"
        / f"fluxes_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(extracted_bitmaps, expected, atol=5e-4, rtol=5e-4)

