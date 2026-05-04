import pathlib

import paint.util.paint_mappings as paint_mappings
import pytest
import torch

from artist import ARTIST_ROOT
from artist.io.paint_calibration_parser import PaintCalibrationDataParser


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
            torch.tensor(
                [50.91342112259258, 6.387824755874856, 87.0], dtype=torch.float64
            ),
            1,
            paint_mappings.UTIS_KEY,
            [
                torch.tensor(
                    [
                        [
                            -17.639921188354,
                            -2.744207382202,
                            50.708946228027,
                            1.000000000000,
                        ]
                    ]
                ),
                torch.tensor(
                    [[0.881544291973, 0.072294861078, -0.466533094645, 0.000000000000]]
                ),
                torch.tensor([[16963.0, 72374.0]]),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0]),
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
            torch.tensor(
                [50.91342112259258, 6.387824755874856, 87.0], dtype=torch.float64
            ),
            3,
            paint_mappings.HELIOS_KEY,
            [
                torch.tensor(
                    [
                        [
                            -17.614156723022,
                            -2.744694471359,
                            50.708179473877,
                            1.000000000000,
                        ],
                        [
                            -17.210176467896,
                            -2.746782302856,
                            51.339900970459,
                            1.000000000000,
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            0.881544291973,
                            0.072294861078,
                            -0.466533094645,
                            0.000000000000,
                        ],
                        [
                            -0.587483644485,
                            0.301079303026,
                            -0.751142024994,
                            0.000000000000,
                        ],
                    ]
                ),
                torch.tensor([[16963.0, 72374.0], [20634.0, 40816.0]]),
                torch.tensor([0, 2], dtype=torch.int32),
                torch.tensor([0, 0]),
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
            torch.tensor(
                [50.91342112259258, 6.387824755874856, 87.0], dtype=torch.float64
            ),
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
                torch.tensor([0, 2], dtype=torch.int32),
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
    sample_limit : int
        Number of samples to be loaded.
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
            PaintCalibrationDataParser(
                sample_limit=sample_limit,
                centroid_extraction_method=centroid_extraction_method,
            )
        assert "not yet supported" in str(exc_info.value)
        assert paint_mappings.UTIS_KEY in str(exc_info.value)
        assert paint_mappings.HELIOS_KEY in str(exc_info.value)
        return
    calibration_data_parser = PaintCalibrationDataParser(
        sample_limit=sample_limit,
        centroid_extraction_method=centroid_extraction_method,
    )
    extracted_list = list(
        calibration_data_parser._parse_calibration_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            heliostat_names=["AA31", "AA39"],
            target_name_to_index={
                "multi_focus_tower": 0,
                "solar_tower_juelich_lower": 1,
                "solar_tower_juelich_upper": 2,
                "receiver": 3,
            },
            power_plant_position=power_plant_position.to(device),
            device=device,
        )
    )

    assert len(expected_list) == len(extracted_list)

    for actual, expected in zip(extracted_list, expected_list):
        torch.testing.assert_close(actual, expected.to(device), atol=5e-4, rtol=5e-4)
