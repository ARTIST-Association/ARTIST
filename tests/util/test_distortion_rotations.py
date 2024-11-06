import math

import pytest
import torch

from artist.util.utils import rotate_distortions


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)


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
    distorted_rays = (
        rotate_distortions(
            e=e_distortions.to(device), u=u_distortions.to(device), device=device
        )
        @ rays_to_rotate.to(device).unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(distorted_rays, expected_distorted_rays.to(device))
