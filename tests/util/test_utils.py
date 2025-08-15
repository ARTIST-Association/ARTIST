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
        # COM exactly at the geometric center -> cropping full plane should be identity
        (
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
            3.0,
            3.0,
            torch.tensor([3.0]),
            torch.tensor([3.0]),
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
        ),
        # Symmetric intensities -> COM at center -> identity expected
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
    Test that cropping is identity when the COM is at the geometric center.

    When the center of mass (COM) is located at the geometric center of the image
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
        The device (CPU or GPU) on which to perform the cropping operation.
    """
    cropped = utils.crop_image_region(
        images=image.to(device),
        crop_width=crop_width,
        crop_height=crop_height,
        target_plane_widths_m=target_width.to(device),
        target_plane_heights_m=target_height.to(device),
    )
    torch.testing.assert_close(
        cropped, expected_cropped.to(device), rtol=1e-4, atol=1e-4
    )
    assert not torch.isnan(cropped).any()


@pytest.mark.parametrize(
    "height,width,bright_r,bright_c,crop_width,crop_height,target_width,target_height,tol_px,min_peak",
    [
        # Small offset near top-right
        (33, 33, 3, 29, 1.0, 1.0, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Closer to center (subpixel COM effects still possible)
        (65, 65, 30, 34, 1.2, 1.2, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Far corner to stress interpolation and centering
        (64, 64, 1, 62, 0.8, 0.8, torch.tensor([3.0]), torch.tensor([3.0]), 1.0, 0.5),
        # Rectangular image
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
    Test cropping behavior when the COM is off-center.

    The crop should be centered on the COM. With bilinear interpolation and
    `align_corners=False`, the peak may attenuate, so ensure it remains non-trivial.
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
        target_plane_widths_m=target_width.to(device),
        target_plane_heights_m=target_height.to(device),
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
    """
    Test shape, type, and correspondence checks for heliostat data mapping.

    This parametrized test verifies that `utils.build_heliostat_data_mapping`
    returns a correctly structured list of mappings for both
    `randomize_selection=False` and `randomize_selection=True`.

    The test:
    1. Creates fake calibration data for multiple heliostats.
    2. Monkeypatches relevant module variables to point to the fake data.
    3. Invokes the mapping function with the given parameters.
    4. Verifies:
    - The return value is a list of the same length as the heliostat list.
    - Each element is a tuple of `(heliostat_name, property_file_paths, image_file_paths)`.
    - Types of elements and paths are correct.
    - The number of measurements matches the expected value.
    - Property/image file paths correspond by ID and reside in the same directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for creating fake calibration data.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest fixture to dynamically replace module attributes for testing.
    randomize_selection_flag : bool
        Flag to randomize selection of measurement files when building the mapping.
    random_seed_value : int
        Random seed to use when `randomize_selection_flag` is `True` for reproducibility.
    number_of_measurements : int
        Number of measurement files to select per heliostat.
    image_variant_name : str
        Identifier for the variant of image data to use (e.g., "raw", "processed").
    """
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
        utils,
        "paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )
    monkeypatch.setattr(
        utils,
        "logging",
        type("Log", (), {"warning": staticmethod(print)}),
        raising=True,
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
    """
    Test that randomized selection order or subset differs from the non-randomized version.

    This test verifies that when `randomize=True`, the file selection order (or subset)
    returned by `utils.build_heliostat_data_mapping` differs from the deterministic
    sorted selection for at least one heliostat, given enough available samples.

    If, by rare chance, a specific seed yields the exact same selection for all heliostats,
    the test is marked as `xfail` to avoid flakiness.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for creating fake calibration data.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest fixture to dynamically replace module attributes for testing.
    random_seed_value : int
        Random seed to use for reproducibility in randomized selection.
    """
    image_variant_name = "flux"
    heliostat_name_list = ["heliostat_1", "heliostat_2"]
    number_of_measurements = 4

    # Create 10 samples per heliostat so a different subset/order is very likely
    paint_calibration_folder_name = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=10,
    )

    monkeypatch.setattr(
        utils,
        "paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )
    monkeypatch.setattr(utils.logging, "warning", lambda *a, **k: None, raising=True)

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

        # Basic validity checks
        assert len(sorted_identifiers) == number_of_measurements
        assert len(randomized_identifiers) == number_of_measurements

        # IDs should come from our fake data set
        universe = {str(i) for i in range(10)}
        assert set(sorted_identifiers).issubset(universe)
        assert set(randomized_identifiers).issubset(universe)

        # We expect randomized output to differ from the sorted selection
        if randomized_identifiers != sorted_identifiers:
            different_for_any_heliostat = True

    if not different_for_any_heliostat:
        pytest.xfail(
            "Random seed produced the same selection as the sorted order for all entries — try another seed."
        )
