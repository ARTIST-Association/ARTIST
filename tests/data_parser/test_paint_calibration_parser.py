import pathlib

import paint.util.paint_mappings as paint_mappings
import pytest
import torch

from artist import ARTIST_ROOT
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser


@pytest.mark.parametrize(
    "heliostat_calibration_mapping, power_plant_position, sample_limit, centroid_extraction_method, expected_list",
    [
        (
            [
                (
                    "AA39",
                    [
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_1.json",
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_2.json",
                    ],
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            1,
            paint_mappings.UTIS_KEY,
            [
                torch.tensor(
                    [
                        [
                            0.180133327842,
                            -3.419259548187,
                            35.798927307129,
                            1.000000000000,
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            -0.094675041735,
                            0.492933481932,
                            -0.864900708199,
                            0.000000000000,
                        ],
                    ]
                ),
                torch.tensor([[28061.0, 47874.0]]),
                torch.tensor([0, 1]),
                torch.tensor([3]),
            ],
        ),
        (
            [
                (
                    "AA39",
                    [
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_1.json",
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_2.json",
                    ],
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            3,
            paint_mappings.HELIOS_KEY,
            [
                torch.tensor(
                    [
                        [0.0901, -3.4193, 35.8465, 1.0000],
                        [-17.4429, -3.0393, 51.5746, 1.0000],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0947, 0.4929, -0.8649, 0.0000],
                        [-0.2741, 0.4399, -0.8552, 0.0000],
                    ]
                ),
                torch.tensor([[28061.0, 47874.0], [22585.0, 48224.0]]),
                torch.tensor([0, 2]),
                torch.tensor([3, 0]),
            ],
        ),
        (
            [
                (
                    "AA39",
                    [
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_1.json",
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_2.json",
                    ],
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            2,
            "invalid",
            [
                torch.tensor(
                    [
                        [0.0901, -3.4193, 35.8465, 1.0000],
                        [-17.4429, -3.0393, 51.5746, 1.0000],
                    ]
                ),
                torch.tensor(
                    [
                        [-0.0947, 0.4929, -0.8649, 0.0000],
                        [-0.2741, 0.4399, -0.8552, 0.0000],
                    ]
                ),
                torch.tensor([[28061.0, 47874.0], [22585.0, 48224.0]]),
                torch.tensor([0, 2]),
                torch.tensor([3, 0]),
            ],
        ),
    ],
)
def test_extract_paint_calibration_data(
    heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
    power_plant_position: torch.Tensor,
    sample_limit: int,
    centroid_extraction_method: str,
    expected_list: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the function to extract calibration data from ``PAINT`` calibration data.

    Parameters
    ----------
    heliostat_calibration_mapping : list[tuple[str, list[pathlib.Path]]]
        The mapping of heliostats and their calibration data files.
    power_plant_position : torch.Tensor
        The power plant position.
    centroid_extraction_method : str
        The centroid extraction method to use.
    expected_list : list[torch.Tensor]
        The expected extracted data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    if centroid_extraction_method == "invalid":
        with pytest.raises(ValueError) as exc_info:
            calibration_data_parser = PaintCalibrationDataParser(
                sample_limit=sample_limit,
                centroid_extraction_method=centroid_extraction_method,
            )
            assert (
                f"The selected centroid extraction method {centroid_extraction_method} is not yet supported. Please use either {paint_mappings.UTIS_KEY} or {paint_mappings.HELIOS_KEY}!"
                in str(exc_info.value)
            )
    else:
        calibration_data_parser = PaintCalibrationDataParser(
            sample_limit=sample_limit,
            centroid_extraction_method=centroid_extraction_method,
        )
        extracted_list = list(
            calibration_data_parser._parse_calibration_data(
                heliostat_calibration_mapping=heliostat_calibration_mapping,
                power_plant_position=power_plant_position.to(device),
                heliostat_names=["AA31", "AA39"],
                target_area_names=[
                    "multi_focus_tower",
                    "receiver",
                    "solar_tower_juelich_upper",
                    "solar_tower_juelich_lower",
                ],
                device=device,
            )
        )

        for actual, expected in zip(extracted_list, expected_list):
            torch.testing.assert_close(
                actual, expected.to(device), atol=5e-4, rtol=5e-4
            )
