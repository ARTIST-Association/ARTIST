import math
import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, utils


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


def test_normalize_bitmaps(device: torch.device) -> None:
    """
    Test the normalization for bitmaps.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    bitmaps_path = (
        pathlib.Path(ARTIST_ROOT)
        / f"tests/data/expected_bitmaps_integration/test_scenario_paint_single_heliostat_{device.type}.pt"
    )

    bitmaps = torch.load(bitmaps_path, map_location=device, weights_only=True)

    normalized_bitmaps = utils.normalize_bitmaps(
        flux_distributions=bitmaps,
        target_area_widths=torch.full(
            (bitmaps.shape[0],),
            config_dictionary.utis_target_width,
            device=device,
        ),
        target_area_heights=torch.full(
            (bitmaps.shape[0],),
            config_dictionary.utis_target_height,
            device=device,
        ),
        number_of_rays=bitmaps.sum(dim=[1, 2]),
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_normalized_bitmaps"
        / f"bitmaps_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(normalized_bitmaps, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "image, crop_width, crop_height, target_width, target_height, expected_cropped",
    [
        # Center of mass exactly at the geometric center -> cropping full plane should be identity.
        (
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
            3.0,
            3.0,
            torch.tensor([3.0]),
            torch.tensor([3.0]),
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
        ),
        # Symmetric intensities -> Center of mass at center -> identity expected.
        (
            torch.tensor([[[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]]]),
            3.0,
            3.0,
            torch.tensor([3.0]),
            torch.tensor([3.0]),
            torch.tensor([[[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]]]),
        ),
    ],
)
def test_crop_image_region_centering(
    image: torch.Tensor,
    crop_width: float,
    crop_height: float,
    target_width: torch.Tensor,
    target_height: torch.Tensor,
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
        Input image tensor to be cropped. Shape is expected to be
        `(C, H, W)` or `(N, C, H, W)`, where `C` is the number of channels.
    crop_width : float
        Desired crop width in meters.
    crop_height : float
        Desired crop height in meters.
    target_width : torch.Tensor
        Target plane width(s) in meters. Should be broadcastable to the batch
        dimension of `image`.
    target_height : torch.Tensor
        Target plane height(s) in meters. Should be broadcastable to the batch
        dimension of `image`.
    expected_cropped : torch.Tensor
        The expected output image tensor after cropping. Should match the input
        `image` in this test scenario.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
    If test does not complete as expected.
    """
    cropped = utils.crop_image_region(
        images=image.to(device),
        crop_width=crop_width,
        crop_height=crop_height,
        target_plane_widths=target_width.to(device),
        target_plane_heights=target_height.to(device),
    )
    torch.testing.assert_close(
        cropped, expected_cropped.to(device), rtol=1e-4, atol=1e-4
    )
    assert not torch.isnan(cropped).any()


@pytest.mark.parametrize(
    "height,width,bright_r,bright_c,crop_width,crop_height,target_width,target_height,tol_px,min_peak",
    [
        # Small offset near top-right.
        (33, 33, 3, 29, 1.0, 1.0, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Closer to center.
        (65, 65, 30, 34, 1.2, 1.2, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Far corner to stress interpolation and centering.
        (64, 64, 1, 62, 0.8, 0.8, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Rectangular image.
        (48, 96, 5, 90, 1.0, 1.5, torch.tensor([4.0]), torch.tensor([2.0]), 1.0, 0.5),
    ],
    ids=[
        "33x33_top-right",
        "65x65_near-center",
        "64x64_far-corner",
        "48x96_rectangular",
    ],
)
def test_crop_image_region_offcenter(
    height: int,
    width: int,
    bright_r: int,
    bright_c: int,
    crop_width: float,
    crop_height: float,
    target_width: torch.Tensor,
    target_height: torch.Tensor,
    tol_px: float,
    min_peak: float,
    device: torch.device,
) -> None:
    """
    Test cropping behavior when the center of mass is off-center.

    This parametrized test verifies that `utils.crop_image_region` correctly centers
    the crop on the image’s center of mass (center of mass) when the bright pixel is not located
    at the geometric center. It also checks that the peak pixel intensity remains
    non-trivial after bilinear interpolation with `align_corners=False`.

    The test:
    1. Builds a synthetic image with a single bright pixel at a specified position.
    2. Crops the image using given crop and target plane dimensions.
    3. Asserts:
    - Output shape matches the input `(C, H, W)`.
    - No NaN values are present.
    - The brightest pixel lies within a specified pixel tolerance of the crop center.
    - The peak pixel intensity is above a specified minimum threshold.

    Parameters
    ----------
    height : int
        Height of the input image in pixels.
    width : int
        Width of the input image in pixels.
    bright_r : int
        Row index of the bright pixel before cropping.
    bright_c : int
        Column index of the bright pixel before cropping.
    crop_width : float
        Crop width in meters.
    crop_height : float
        Crop height in meters.
    target_width : torch.Tensor
        Target plane width(s) in meters; broadcastable to the batch.
    target_height : torch.Tensor
        Target plane height(s) in meters; broadcastable to the batch.
    tol_px : float
        Pixel tolerance allowed between the peak pixel position and the geometric center.
    min_peak : float
        Minimum acceptable peak intensity after cropping and interpolation.
    device : torch.device
     The device on which to initialize tensors.

    Raises
    ------
         AssertionError
              If test does not complete as expected.
    """
    # Build image with a single bright pixel
    image = torch.zeros((1, height, width), dtype=torch.float32)
    # Clamp to valid range just in case parameters push to boundary
    br = int(max(0, min(height - 1, bright_r)))
    bc = int(max(0, min(width - 1, bright_c)))
    image[0, br, bc] = 1.0

    cropped = utils.crop_image_region(
        images=image.to(device),
        crop_width=crop_width,
        crop_height=crop_height,
        target_plane_widths=target_width.to(device),
        target_plane_heights=target_height.to(device),
    )

    # Sanity checks
    assert cropped.shape == image.shape[-3:], "Function keeps HxW by contract"
    assert not torch.isnan(cropped).any()

    # Locate the maximum (batch, row, col)
    max_val = torch.amax(cropped)
    pos = torch.nonzero(cropped == max_val, as_tuple=False)[0]
    _, r, c = pos.tolist()
    height_cropped, width_cropped = cropped.shape[-2], cropped.shape[-1]
    center_r = (height_cropped - 1) / 2.0
    center_c = (width_cropped - 1) / 2.0

    # Allow a little extra slack on even dimensions due to half-pixel center with align_corners=False
    tol_r = tol_px + (0.5 if (height_cropped % 2 == 0) else 0.0)
    tol_c = tol_px + (0.5 if (width_cropped % 2 == 0) else 0.0)

    assert abs(r - center_r) <= tol_r, f"max row {r} not centered (H={height_cropped})"
    assert abs(c - center_c) <= tol_c, f"max col {c} not centered (W={width_cropped})"

    assert max_val >= min_peak
