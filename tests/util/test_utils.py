import math
from unittest.mock import MagicMock

import pytest
import torch

from artist.field.solar_tower import SolarTower
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import utils


@pytest.mark.parametrize(
    "east_translation, north_translation, up_translation, expected",
    [
        (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            ),
        ),
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            None,
        ),
    ],
)
def test_translate_enu(
    east_translation: torch.Tensor,
    north_translation: torch.Tensor,
    up_translation: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the correct translation matrix is created.

    Parameters
    ----------
    east_translation : torch.Tensor
        The translation in east direction.
    north_translation : torch.Tensor
        The translation in north direction.
    up_translation : torch.Tensor
        The translation in up direction.
    expected : torch.Tensor
        The expected overall translation or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            utils.translate_enu(
                e=east_translation.to(device),
                n=north_translation.to(device),
                u=up_translation.to(device),
                device=device,
            )
        assert (
            "The three tensors containing the east, north, and up translations must have the same shape."
            in str(exc_info.value)
        )
    else:
        # Check if the translation matrix is correct.
        translation_matrix = utils.translate_enu(
            e=east_translation.to(device),
            n=north_translation.to(device),
            u=up_translation.to(device),
            device=device,
        )
        torch.testing.assert_close(
            translation_matrix, expected.to(device), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "point, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
        ),
        (
            torch.tensor([1.0, 0.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            None,
        ),
    ],
)
def test_3d_point_converter(
    point: torch.Tensor, expected: torch.Tensor | None, device: torch.device
) -> None:
    """
    Test the 3d to 4d point converter.

    Parameters
    ----------
    point : torch.Tensor
        A 3d point.
    expected : torch.Tensor | None
        A 4d point or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            utils.convert_3d_points_to_4d_format(
                points=point.to(device),
                device=device,
            )
        assert f"Expected 3D points but got points of shape {point.shape}!" in str(
            exc_info.value
        )
    else:
        # Check if the 4d point is correct.
        point_4d = utils.convert_3d_points_to_4d_format(
            points=point.to(device),
            device=device,
        )
        torch.testing.assert_close(point_4d, expected.to(device), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "direction, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ),
        (
            torch.tensor([1.0, 0.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            None,
        ),
    ],
)
def test_3d_direction_converter(
    direction: torch.Tensor, expected: torch.Tensor | None, device: torch.device
) -> None:
    """
    Test the 3d to 4d point converter.

    Parameters
    ----------
    direction : torch.Tensor
        A 3d direction.
    expected : torch.Tensor | None
        A 4d direction or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            utils.convert_3d_directions_to_4d_format(
                directions=direction.to(device),
                device=device,
            )
        assert (
            f"Expected 3D directions but got directions of shape {direction.shape}!"
            in str(exc_info.value)
        )
    else:
        # Check if the 4d point is correct.
        direction_4d = utils.convert_3d_directions_to_4d_format(
            directions=direction.to(device),
            device=device,
        )
        torch.testing.assert_close(
            direction_4d, expected.to(device), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "e_distortions, u_distortions, rays_to_rotate, expected_distorted_rays",
    [
        (  # Rotate single ray in the east direction.
            torch.tensor([[[math.pi / 2]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[[1.0, -1.0, 1.0, 0.0]]]]),
        ),
        (  # Rotate single ray in the up direction.
            torch.tensor([[[0.0]]]),
            torch.tensor([[[math.pi / 2]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[[-1.0, 1.0, 1.0, 0.0]]]]),
        ),
        (  # Rotate single ray in the east and then up direction.
            torch.tensor([[[math.pi / 2]]]),
            torch.tensor([[[math.pi / 2]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[[-1.0, -1.0, 1.0, 0.0]]]]),
        ),
        (  # Consider multiple rotations for a single ray in the east direction.
            torch.tensor(
                [[[math.pi / 2]], [[math.pi]], [[math.pi / 4]], [[2 * math.pi]]]
            ),
            torch.tensor([[[0]], [[0]], [[0]], [[0]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor(
                [
                    [[[1.000000, -1.000000, 1.000000, 0.000000]]],
                    [[[1.000000, -1.000000, -1.000000, 0.000000]]],
                    [[[1.000000, 0.000000, 1.414214, 0.000000]]],
                    [[[1.000000, 1.000000, 1.000000, 0.000000]]],
                ]
            ),
        ),
        (  # Consider multiple rotations for a single ray in the up direction.
            torch.tensor([[[0]], [[0]], [[0]], [[0]]]),
            torch.tensor(
                [[[math.pi / 2]], [[math.pi]], [[math.pi / 4]], [[2 * math.pi]]]
            ),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor(
                [
                    [[[-1.000000, 1.000000, 1.000000, 0.000000]]],
                    [[[-1.000000, -1.000000, 1.000000, 0.000000]]],
                    [[[0.000000, 1.414214, 1.000000, 0.000000]]],
                    [[[1.000000, 1.000000, 1.000000, 0.000000]]],
                ]
            ),
        ),
        (  # Consider multiple rotations for a single ray in the east and then up direction.
            torch.tensor(
                [[[math.pi / 2]], [[math.pi]], [[math.pi / 4]], [[2 * math.pi]]]
            ),
            torch.tensor(
                [[[math.pi / 2]], [[math.pi]], [[math.pi / 4]], [[2 * math.pi]]]
            ),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor(
                [
                    [[[-1.000000, -1.000000, 1.000000, 0.000000]]],
                    [[[-1.000000, 1.000000, -1.000000, 0.000000]]],
                    [[[0.000000, 0.292893, 1.707107, 0.000000]]],
                    [[[1.000000, 1.000000, 1.000000, 0.000000]]],
                ]
            ),
        ),
        (  # Consider multiple rotations for multiple rays in the east direction.
            torch.tensor(
                [
                    [[math.pi, math.pi / 2, math.pi / 4]],
                    [[math.pi / 4, math.pi, math.pi / 2]],
                ]
            ),
            torch.tensor([[[0, 0, 0]], [[0, 0, 0]]]),
            torch.tensor(
                [[[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]]]
            ),
            torch.tensor(
                [
                    [
                        [
                            [1.000000e00, -9.999999e-01, -1.000000e00, 0.000000e00],
                            [2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [3.000000e00, -5.960464e-08, 4.242640e00, 0.000000e00],
                        ]
                    ],
                    [
                        [
                            [1.000000e00, 0.000000e00, 1.414214e00, 0.000000e00],
                            [2.000000e00, -2.000000e00, -2.000000e00, 0.000000e00],
                            [3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ]
                    ],
                ]
            ),
        ),
        (  # Consider multiple rotations for multiple rays in the up direction.
            torch.tensor([[[0, 0, 0]], [[0, 0, 0]]]),
            torch.tensor(
                [
                    [[math.pi, math.pi / 2, math.pi / 4]],
                    [[math.pi / 4, math.pi, math.pi / 2]],
                ]
            ),
            torch.tensor(
                [[[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]]]
            ),
            torch.tensor(
                [
                    [
                        [
                            [-9.999999e-01, -1.000000e00, 1.000000e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 4.242640e00, 3.000000e00, 0.000000e00],
                        ]
                    ],
                    [
                        [
                            [0.000000e00, 1.414214e00, 1.000000e00, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-3.000000e00, 3.000000e00, 3.000000e00, 0.000000e00],
                        ]
                    ],
                ]
            ),
        ),
        (  # Consider multiple rotations for multiple rays in the east and then up direction.
            torch.tensor(
                [
                    [[math.pi, math.pi / 2, math.pi / 4]],
                    [[math.pi / 4, math.pi, math.pi / 2]],
                ]
            ),
            torch.tensor(
                [
                    [[math.pi, math.pi / 2, math.pi / 4]],
                    [[math.pi / 4, math.pi, math.pi / 2]],
                ]
            ),
            torch.tensor(
                [[[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]]]
            ),
            torch.tensor(
                [
                    [
                        [
                            [-9.999999e-01, 1.000000e00, -9.999999e-01, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 8.786795e-01, 5.121320e00, 0.000000e00],
                        ]
                    ],
                    [
                        [
                            [0.000000e00, 2.928932e-01, 1.707107e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, -2.000000e00, 0.000000e00],
                            [-3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ]
                    ],
                ]
            ),
        ),
        (  # Consider multiple rotations for multiple rays on four facets in the east and then up direction.
            torch.tensor(
                [
                    [
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                    ],
                    [
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                        [math.pi, math.pi / 2, math.pi / 4],
                    ],
                    [
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                        [math.pi / 4, math.pi, math.pi / 2],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
                    [[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
                    [[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
                    [[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            [-9.999999e-01, 1.000000e00, -9.999999e-01, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 8.786795e-01, 5.121320e00, 0.000000e00],
                        ],
                        [
                            [-9.999999e-01, 1.000000e00, -9.999999e-01, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 8.786795e-01, 5.121320e00, 0.000000e00],
                        ],
                        [
                            [-9.999999e-01, 1.000000e00, -9.999999e-01, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 8.786795e-01, 5.121320e00, 0.000000e00],
                        ],
                        [
                            [-9.999999e-01, 1.000000e00, -9.999999e-01, 0.000000e00],
                            [-2.000000e00, -2.000000e00, 2.000000e00, 0.000000e00],
                            [-5.960464e-08, 8.786795e-01, 5.121320e00, 0.000000e00],
                        ],
                    ],
                    [
                        [
                            [0.000000e00, 2.928932e-01, 1.707107e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, -2.000000e00, 0.000000e00],
                            [-3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ],
                        [
                            [0.000000e00, 2.928932e-01, 1.707107e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, -2.000000e00, 0.000000e00],
                            [-3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ],
                        [
                            [0.000000e00, 2.928932e-01, 1.707107e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, -2.000000e00, 0.000000e00],
                            [-3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ],
                        [
                            [0.000000e00, 2.928932e-01, 1.707107e00, 0.000000e00],
                            [-2.000000e00, 2.000000e00, -2.000000e00, 0.000000e00],
                            [-3.000000e00, -3.000000e00, 3.000000e00, 0.000000e00],
                        ],
                    ],
                ]
            ),
        ),
        (  # Test raise ValueError
            torch.tensor([[[math.pi / 2]]]),
            torch.tensor([[[0.0], [0.0]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            None,
        ),
    ],
)
def test_distortion_rotations(
    e_distortions: torch.Tensor,
    u_distortions: torch.Tensor,
    rays_to_rotate: torch.Tensor,
    expected_distorted_rays: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the rotation function used for scattering rays by considering various rotations.

    Parameters
    ----------
    e_distortions : torch.Tensor
        The distortions in the east direction used in the rotation matrix.
    u_distortions : torch.Tensor
        The distortions in the upper direction used in the rotation matrix.
    rays_to_rotate : torch.Tensor
        The rays to rotate given the distortions.
    expected_distorted_rays : torch.Tensor
        The expected distorted rays after rotation.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    if expected_distorted_rays is None:
        with pytest.raises(ValueError) as exc_info:
            distorted_rays = (
                utils.rotate_distortions(
                    e=e_distortions.to(device),
                    u=u_distortions.to(device),
                    device=device,
                )
                @ rays_to_rotate.to(device).unsqueeze(-1)
            ).squeeze(-1)
        assert (
            "The two tensors containing angles for the east and up rotation must have the same shape."
            in str(exc_info.value)
        )
    else:
        distorted_rays = (
            utils.rotate_distortions(
                e=e_distortions.to(device), u=u_distortions.to(device), device=device
            )
            @ rays_to_rotate.to(device).unsqueeze(-1)
        ).squeeze(-1)

        torch.testing.assert_close(distorted_rays, expected_distorted_rays.to(device))


@pytest.mark.parametrize(
    "total_width, slope_width, plateau_width, expected",
    [
        (8, 2, 4, torch.tensor([0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.25])),
        (4, 2, 4, torch.tensor([1.0, 1.0, 1.0, 1.0])),
        (1, 2, 3, torch.tensor([1.0])),
        (
            10,
            2,
            2,
            torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 0.75, 0.25, 0.0, 0.0]),
        ),
    ],
)
def test_trapezoid_distribution(
    total_width: int,
    slope_width: int,
    plateau_width: int,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the trapezoid distribution works as expected.

    Parameters
    ----------
    total_width : int
        The total width of the trapezoid.
    slope_width : int
        The width of the slope of the trapezoid.
    plateau_width : int
        The width of the plateau.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    trapezoid = utils.trapezoid_distribution(
        total_width=total_width,
        slope_width=slope_width,
        plateau_width=plateau_width,
        device=device,
    )

    torch.testing.assert_close(trapezoid, expected.to(device), atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "image, crop_width, crop_height, target_area_indices, expected_cropped",
    [
        # Symmetric bitmaps and no change in dimensions.
        (
            torch.tensor(
                [
                    [[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]],
                    [[0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 0.0, 0.5]],
                ]
            ),
            3.0,
            3.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]],
                    [[0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 0.0, 0.5]],
                ]
            ),
        ),
        # Symmetric bitmaps and change in dimensions.
        (
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 2.0, 1.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [1.0, 2.0, 2.0, 1.0],
                    ],
                    [
                        [1.0, 2.0, 2.0, 1.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [1.0, 2.0, 2.0, 1.0],
                    ],
                ]
            ),
            5.0,
            5.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000],
                    ],
                ]
            ),
        ),
        # Asymmetric bitmaps and no change in dimensions.
        (
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ]
            ),
            3.0,
            3.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        ),
        # Asymmetric bitmaps and change in dimensions.
        (
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            2.0,
            2.0,
            torch.tensor([0]),
            torch.tensor(
                [
                    [
                        [0.1111, 0.3333, 0.3333, 0.3333, 0.3333, 0.1111],
                        [0.3333, 1.0000, 1.0000, 1.0000, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.4444, 1.4444, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.4444, 1.4444, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.0000, 1.0000, 1.0000, 0.3333],
                        [0.1111, 0.3333, 0.3333, 0.3333, 0.3333, 0.1111],
                    ]
                ]
            ),
        ),
    ],
)
def test_crop_flux_distributions_around_center_centering(
    image: torch.Tensor,
    crop_width: float,
    crop_height: float,
    target_area_indices: torch.Tensor,
    expected_cropped: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that cropping is identity when the center of mass is at the geometric center.

    When the center of mass is located at the geometric center of the image
    and the crop dimensions span the full target plane, the cropping operation
    should return the image unchanged.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor to be cropped.
        Tensor of shape [number_of_bitmaps, bitmap_resolution_e, bitmap_resolution_u].
    crop_width : float
        Desired crop width in meters.
    crop_height : float
        Desired crop height in meters.
    expected_cropped : torch.Tensor
        The expected output image tensor after cropping.
        Tensor of shape [number_of_bitmaps, bitmap_resolution_e, bitmap_resolution_u].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_solar_tower = MagicMock(spec=SolarTower)
    mock_target_areas_planar = MagicMock(spec=TowerTargetAreasPlanar)
    mock_target_areas_planar.names = ["multi_focus_tower"]
    mock_target_areas_planar.dimensions = torch.tensor([[3.0, 3.0]], device=device)
    mock_target_areas_cylindrical = MagicMock(spec=TowerTargetAreasCylindrical)
    mock_target_areas_cylindrical.names = ["receiver"]
    mock_target_areas_cylindrical.radii = torch.tensor(([[1.0]]), device=device)
    mock_target_areas_cylindrical.heights = torch.tensor(([[3.0]]), device=device)
    mock_target_areas_cylindrical.opening_angles = torch.tensor(
        ([[3.0]]), device=device
    )

    mock_solar_tower.target_areas = [
        mock_target_areas_planar,
        mock_target_areas_cylindrical,
    ]
    mock_solar_tower.number_of_target_area_types = 2
    mock_solar_tower.number_of_target_areas_per_type = torch.tensor(
        [1, 1], device=device
    )
    mock_solar_tower.target_name_to_index = {"multi_focus_tower": 0, "receiver": 1}
    mock_solar_tower.index_to_target_area = {0: "multi_focus_tower", 1: "receiver"}

    cropped = utils.crop_flux_distributions_around_center(
        flux_distributions=image.to(device),
        solar_tower=mock_solar_tower,
        target_area_indices=target_area_indices.to(device),
        crop_width=crop_width,
        crop_height=crop_height,
        device=device,
    )
    torch.testing.assert_close(
        cropped, expected_cropped.to(device), rtol=1e-4, atol=1e-4
    )
    assert not torch.isnan(cropped).any()


@pytest.mark.parametrize(
    "wgs84_coordinates, reference_point, expected_enu_coordinates",
    [
        # Coordinates of Juelich power plant and multi-focus tower.
        (
            (
                torch.tensor(
                    [[50.91339645088695, 6.387574436728054, 138.97975]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [50.913421630859, 6.387824755874856, 87.000000000000],
                    dtype=torch.float64,
                ),
                torch.tensor([[-17.6045, -2.8012, 51.9798]]),
            )
        ),
    ],
)
def test_wgs84_to_enu_converter(
    wgs84_coordinates: torch.Tensor,
    reference_point: torch.Tensor,
    expected_enu_coordinates: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the WGS84 to ENU conversion.

    Parameters
    ----------
    wgs84_coordinates : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    expected_enu_coordinates : torch.Tensor
        The expected enu coordinates.
    device : torch.device| str
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    calculated_enu_coordinates = utils.convert_wgs84_coordinates_to_local_enu(
        wgs84_coordinates.to(device), reference_point.to(device), device
    )

    torch.testing.assert_close(
        calculated_enu_coordinates, expected_enu_coordinates.to(device)
    )


@pytest.mark.parametrize(
    "azimuth, elevation, degree, expected",
    [
        (
            torch.tensor([-45.0, -45.0, 45.0, 135.0, 225.0, 315.0]),
            torch.tensor([0.0, 45.0, 45.0, 45.0, 45.0, 45.0]),
            True,
            torch.tensor(
                [
                    [
                        -1 / torch.sqrt(torch.tensor([2.0])),
                        -1 / torch.sqrt(torch.tensor([2.0])),
                        0.0,
                    ],
                    [
                        -0.5,
                        -0.5,
                        1 / torch.sqrt(torch.tensor([2.0])),
                    ],
                    [0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [0.5, 0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [-0.5, 0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [-0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                ]
            ),
        ),
        (
            torch.tensor([-torch.pi / 4, torch.pi / 4]),
            torch.tensor([torch.pi / 4, torch.pi / 4]),
            False,
            torch.tensor(
                [
                    [-0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                ]
            ),
        ),
    ],
)
def test_azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    degree: bool,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the azimuth, elevation to east, north, up converter.

    Parameters
    ----------
    azimuth : torch.Tensor
        The azimuth angle.
    elevation : torch.Tensor
        The elevation angle.
    degree : bool
        Angles in degree.
    expected : torch.Tensor
        The expected coordinates in the ENU (east, north, up) coordinate system.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    enu_coordinates = utils.azimuth_elevation_to_enu(
        azimuth=azimuth, elevation=elevation, degree=degree, device=device
    )
    torch.testing.assert_close(
        enu_coordinates, expected.to(device), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "from_orientation, to_orientation, expected_axis, expected_angle",
    [
        # Same orientation, no rotation, zero degree angle.
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0]),
        ),
        # From x-axis to y-axis, 90 degrees rotation around z.
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([torch.pi / 2]),
        ),
        # From y-axis to z-axis, 90 degrees rotation around x.
        (
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([torch.pi / 2]),
        ),
        # From positive x-axis, ti negative x-axis, 180 degrees rotation, .
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([torch.pi]),
        ),
        # Non-normalized input vectors, from x-axis to y-axis, 90 degrees.
        (
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 3.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([torch.pi / 2]),
        ),
    ],
)
def test_rotation_angle_and_axis(
    from_orientation: torch.Tensor,
    to_orientation: torch.Tensor,
    expected_axis: torch.Tensor,
    expected_angle: float,
    device: torch.device,
):
    axis, angle = utils.rotation_angle_and_axis(
        from_orientation=from_orientation.to(device),
        to_orientation=to_orientation.to(device),
        device=device,
    )

    assert torch.allclose(axis, expected_axis.to(device), atol=1e-5)
    assert torch.isclose(angle, expected_angle.to(device), atol=1e-5)


@pytest.mark.parametrize(
    "bitmap_coordinates, bitmap_resolution, target_area_indices, expected_coordinates",
    [
        (
            torch.tensor([[127.5, 127.5], [63.75, 255.0], [0.0, 0.0]]),
            torch.tensor([256, 256]),
            torch.tensor([0, 0, 1]),
            torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [1.5, 0.0, -3.0, 1.0], [2.0, 0.0, 4.0, 1.0]]
            ),
        ),
        (
            torch.tensor([[127.5, 127.5], [127.5, 255.0], [0.0, 63.75]]),
            torch.tensor([256, 256]),
            torch.tensor([2, 2, 2]),
            torch.tensor(
                [[0.0, 2.0, 0.0, 1.0], [0.0, 2.0, -3.0, 1.0], [2.0, 0.0, 1.5, 1.0]]
            ),
        ),
        (
            torch.tensor([[255.0, 191.25], [255.0, 255.0]]),
            torch.tensor([256, 256]),
            torch.tensor([2, 0]),
            torch.tensor([[-2.0, 0.0, -1.5, 1.0], [-3.0, 0.0, -3.0, 1.0]]),
        ),
    ],
)
def test_bitmap_coordinates_to_target_coordinates(
    bitmap_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    target_area_indices: torch.Tensor,
    expected_coordinates: torch.Tensor,
    device: torch.device,
):
    mock_solar_tower = MagicMock(spec=SolarTower)
    mock_target_areas_planar = MagicMock(spec=TowerTargetAreasPlanar)
    mock_target_areas_planar.names = ["planar1", "planar2"]
    mock_target_areas_planar.dimensions = torch.tensor(
        [[6.0, 6.0], [2.0, 4.0]], device=device
    )
    mock_target_areas_planar.centers = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 2.0, 1.0]], device=device
    )
    mock_target_areas_cylindrical = MagicMock(spec=TowerTargetAreasCylindrical)
    mock_target_areas_cylindrical.names = ["cylinder1"]
    mock_target_areas_cylindrical.normals = torch.tensor(
        ([[0.0, 1.0, 0.0, 0.0]]), device=device
    )
    mock_target_areas_cylindrical.axes = torch.tensor(
        ([[0.0, 0.0, 1.0, 0.0]]), device=device
    )
    mock_target_areas_cylindrical.radii = torch.tensor(([[2.0]]), device=device)
    mock_target_areas_cylindrical.heights = torch.tensor(([[6.0]]), device=device)
    mock_target_areas_cylindrical.opening_angles = torch.tensor(
        ([[math.pi]]), device=device
    )
    mock_target_areas_cylindrical.centers = torch.tensor(
        ([[0.0, 0.0, 0.0, 1.0]]), device=device
    )

    mock_solar_tower.target_areas = [
        mock_target_areas_planar,
        mock_target_areas_cylindrical,
    ]
    mock_solar_tower.number_of_target_area_types = 2
    mock_solar_tower.number_of_target_areas_per_type = torch.tensor(
        [2, 1], device=device
    )
    mock_solar_tower.target_name_to_index = {"planar1": 0, "planar2": 1, "cylinder1": 2}
    mock_solar_tower.index_to_target_area = {0: "planar1", 1: "planar2", 2: "cylinder1"}

    target_coordinates = utils.bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coordinates.to(device),
        bitmap_resolution=bitmap_resolution.to(device),
        solar_tower=mock_solar_tower,
        target_area_indices=target_area_indices.to(device),
        device=device,
    )

    torch.testing.assert_close(
        target_coordinates, expected_coordinates.to(device), rtol=1e-4, atol=1e-4
    )
