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
    "image, crop_w, crop_h, target_w, target_h, expected_cropped",
    [
        (
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
            1.0,
            1.0,
            torch.tensor([3.0]),
            torch.tensor([3.0]),
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
        ),
        (
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
            1.0,
            1.0,
            torch.tensor([3.0]),
            torch.tensor([3.0]),
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
        ),
    ],
)
def test_crop_image_region_centering(
    image: torch.Tensor,
    crop_w: float,
    crop_h: float,
    target_w: torch.Tensor,
    target_h: torch.Tensor,
    expected_cropped: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that cropping is centered on the center of mass.

    Parameters
    ----------
    image : torch.Tensor
        A grayscale image or batch of images.
    crop_w : float
        Desired crop width in meters.
    crop_h : float
        Desired crop height in meters.
    target_w : torch.Tensor
        Width of the full target plane in meters.
    target_h : torch.Tensor
        Height of the full target plane in meters.
    expected_cropped : torch.Tensor
        The expected cropped image.
    device : torch.device
        The device on which to run the test.

    Raises
    ------
    AssertionError
        If the output is not as expected.
    """
    cropped = utils.crop_image_region(
        images=image.to(device),
        crop_width=crop_w,
        crop_height=crop_h,
        target_plane_widths_m=target_w.to(device),
        target_plane_heights_m=target_h.to(device),
    )
    # The cropping should be centered on the center of mass, so for these simple cases, output should match input.
    torch.testing.assert_close(
        cropped, expected_cropped.to(device), rtol=1e-4, atol=1e-4
    )
    assert not torch.isnan(cropped).any()


def test_crop_image_region_offcenter(device: torch.device) -> None:
    """
    Test cropping logic for an off-center mass.

    Parameters
    ----------
    device : torch.device
        The device on which to run the test.

    Raises
    ------
    AssertionError
        If the output is not as expected.
    """
    # Center of mass is at (2, 0) in a 3x3 image
    image = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    crop_w = 1.0
    crop_h = 1.0
    target_w = torch.tensor([3.0])
    target_h = torch.tensor([3.0])

    cropped = utils.crop_image_region(
        images=image.to(device),
        crop_width=crop_w,
        crop_height=crop_h,
        target_plane_widths_m=target_w.to(device),
        target_plane_heights_m=target_h.to(device),
    )

    # The maximum value should remain at the same location after cropping
    max_idx = torch.argmax(cropped)
    assert max_idx == 2  # Should be at index 2 (last column)
    assert cropped[0, 0, 2] > 0.9
    assert not torch.isnan(cropped).any()
    assert not torch.isnan(cropped).any()


def _make_fake_calibration_data(
    base_directory_path: pathlib.Path,
    heliostat_name_list,
    image_variant_name: str,
    count_per_heliostat: int,
):
    """Create a deterministic fake folder tree with property/image pairs."""
    paint_calibration_folder_name = "paint_calibration"
    for heliostat_name in heliostat_name_list:
        calibration_directory_path = (
            base_directory_path / heliostat_name / paint_calibration_folder_name
        )
        calibration_directory_path.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_heliostat):
            (
                calibration_directory_path / f"{index}-calibration-properties.json"
            ).write_text("{}")
            # Write a tiny, obviously-fake PNG header so opening as binary won't crash
            (
                calibration_directory_path / f"{index}-{image_variant_name}.png"
            ).write_bytes(b"\x89PNG\r\nfake")
    return paint_calibration_folder_name


@pytest.mark.parametrize(
    "randomize_selection_flag,random_seed_value,number_of_measurements,image_variant_name",
    [
        (False, 0, 2, "flux"),
        (True, 123, 2, "flux"),
    ],
)
def test_build_heliostat_data_mapping_shape_parametrized(
    tmp_path: pathlib.Path,
    monkeypatch,
    randomize_selection_flag,
    random_seed_value,
    number_of_measurements,
    image_variant_name,
):
    """Shape/type/correspondence checks for both randomize_selection=False and randomize_selection=True."""
    heliostat_name_list = ["heliostat_1", "heliostat_2"]
    # Create 5 samples per heliostat so we can select a subset
    paint_calibration_folder_name = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=5,
    )

    # Patch where the function actually reads these names
    monkeypatch.setattr(
        "utils.paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )
    monkeypatch.setattr(
        "utils.log", type("Log", (), {"warning": staticmethod(print)}), raising=True
    )

    result_mapping_list = utils.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        num_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=randomize_selection_flag,
        seed=random_seed_value,
    )

    # --- Shape checks ---
    assert isinstance(result_mapping_list, list)
    assert len(result_mapping_list) == len(heliostat_name_list)

    for heliostat_entry in result_mapping_list:
        assert isinstance(heliostat_entry, tuple) and len(heliostat_entry) == 3
        heliostat_name, property_file_paths, image_file_paths = heliostat_entry

        assert isinstance(heliostat_name, str) and heliostat_name in heliostat_name_list

        assert isinstance(property_file_paths, list)
        assert isinstance(image_file_paths, list)
        assert all(
            isinstance(property_path, pathlib.Path)
            for property_path in property_file_paths
        )
        assert all(
            isinstance(image_path, pathlib.Path) for image_path in image_file_paths
        )

        assert len(property_file_paths) == number_of_measurements
        assert len(image_file_paths) == number_of_measurements

        # Correspondence by ID and directory
        for property_file_path, image_file_path in zip(
            property_file_paths, image_file_paths
        ):
            assert property_file_path.parent == image_file_path.parent
            assert (
                property_file_path.stem.split("-")[0]
                == image_file_path.stem.split("-")[0]
            )


@pytest.mark.parametrize("random_seed_value", [7, 11, 123, 2024])
def test_build_heliostat_data_mapping_randomization_changes_order(
    tmp_path: pathlib.Path,
    monkeypatch,
    random_seed_value,
):
    """Check randomized selection order differs from sorted order for at least one heliostat.

    With enough samples available, the randomized selection order should differ from
    the non-randomized (sorted) order for at least one heliostat (highly likely).
    If a rare seed yields the same order, xfail to avoid flakiness.
    """
    image_variant_name = "flux"
    heliostat_name_list = ["heliostat_1", "heliostat_2"]
    number_of_measurements = 4
    # Create 10 samples per heliostat to make a different order very likely
    paint_calibration_folder_name = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=10,
    )

    # Patch where the function actually reads these names
    monkeypatch.setattr(
        "utils.paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )
    monkeypatch.setattr(
        "utils.log", type("Log", (), {"warning": staticmethod(print)}), raising=True
    )

    result_sorted_list = utils.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        num_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=False,
        seed=random_seed_value,
    )
    result_randomized_list = utils.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        num_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=True,
        seed=random_seed_value,
    )

    # Compare per heliostat
    different_for_any_heliostat = False
    for (sorted_name, sorted_property_paths, _), (
        random_name,
        randomized_property_paths,
        _,
    ) in zip(result_sorted_list, result_randomized_list):
        assert sorted_name == random_name
        sorted_identifiers = [p.stem.split("-")[0] for p in sorted_property_paths]
        randomized_identifiers = [
            p.stem.split("-")[0] for p in randomized_property_paths
        ]
        # Same set, possibly different order
        assert set(sorted_identifiers) == set(randomized_identifiers)
        if sorted_identifiers != randomized_identifiers:
            different_for_any_heliostat = True

    if not different_for_any_heliostat:
        pytest.xfail(
            "Random seed produced the same order for all entries — try another seed."
        )
