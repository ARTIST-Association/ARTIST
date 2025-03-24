from typing import Optional

import pytest
import torch

from artist.util import utils


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
    point: torch.Tensor, expected: Optional[torch.Tensor], device: torch.device
) -> None:
    """
    Test the 3d to 4d point converter.

    Parameters
    ----------
    point : torch.Tensor
        A 3d point.
    expected : Optional[torch.Tensor]
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
            utils.convert_3d_point_to_4d_format(
                point=point.to(device),
                device=device,
            )
        assert f"Expected a 3D point but got a point of shape {point.shape}!" in str(
            exc_info.value
        )
    else:
        # Check if the 4d point is correct.
        point_4d = utils.convert_3d_point_to_4d_format(
            point=point.to(device),
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
    direction: torch.Tensor, expected: Optional[torch.Tensor], device: torch.device
) -> None:
    """
    Test the 3d to 4d point converter.

    Parameters
    ----------
    direction : torch.Tensor
        A 3d direction.
    expected : Optional[torch.Tensor]
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
            utils.convert_3d_direction_to_4d_format(
                direction=direction.to(device),
                device=device,
            )
        assert (
            f"Expected a 3D direction but got a direction of shape {direction.shape}!"
            in str(exc_info.value)
        )
    else:
        # Check if the 4d point is correct.
        direction_4d = utils.convert_3d_direction_to_4d_format(
            direction=direction.to(device),
            device=device,
        )
        torch.testing.assert_close(
            direction_4d, expected.to(device), rtol=1e-4, atol=1e-4
        )
