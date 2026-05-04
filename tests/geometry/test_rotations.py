import pytest
import torch

from artist.geometry import rotations


@pytest.mark.parametrize(
    "from_orientation, to_orientation, expected_axis, expected_angle",
    [
        # Same orientation, no rotation, zero-degree angle.
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(0.0),
        ),
        # From x-axis to y-axis, 90 degrees rotation around z.
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor(torch.pi / 2),
        ),
        # From y-axis to z-axis, 90 degrees rotation around x.
        (
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(torch.pi / 2),
        ),
        # From positive x-axis, to negative x-axis, 180 degrees rotation, .
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor(torch.pi),
        ),
        # Non-normalized input vectors, from x-axis to y-axis, 90 degrees.
        (
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 3.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor(torch.pi / 2),
        ),
    ],
)
def test_rotation_angle_and_axis(
    from_orientation: torch.Tensor,
    to_orientation: torch.Tensor,
    expected_axis: torch.Tensor,
    expected_angle: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the get rotation axis and angle method.

    Parameters
    ----------
    from_orientation : torch.Tensor
        The initial orientation (e.g., a 3D vector or rotation representation).
    to_orientation : torch.Tensor
        The target orientation to rotate into.
    expected_axis : torch.Tensor
        The expected unit vector representing the rotation axis.
    expected_angle : torch.Tensor
        The expected rotation angle in radians.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    axis, angle = rotations.rotation_angle_and_axis(
        from_orientation=from_orientation.to(device),
        to_orientation=to_orientation.to(device),
        device=device,
    )

    assert torch.allclose(axis, expected_axis.to(device), atol=1e-5)
    assert torch.isclose(angle, expected_angle.to(device), atol=1e-5)
