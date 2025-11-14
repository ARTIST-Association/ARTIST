import pytest
import torch

from artist.core.core_utils import scale_loss


@pytest.mark.parametrize(
    "loss, reference, weight, expected",
    [
        (torch.tensor([2.0]), torch.tensor([2.0]), 1.0, torch.tensor([2.0])),
        (torch.tensor([2.0]), torch.tensor([2.0]), 0.5, torch.tensor([1.0])),
        (torch.tensor([4.0]), torch.tensor([2.0]), 1.0, torch.tensor([2.0])),
        (torch.tensor([0.0]), torch.tensor([2.0]), 1.0, torch.tensor([0.0])),
    ],
)
def test_scale_loss(
    loss: torch.Tensor,
    reference: torch.Tensor,
    weight: float,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the scale loss function.

    Parameters
    ----------
    loss : torch.Tensor
        The loss to be scaled.
        Tensor of shape [1].
    reference :  torch.Tensor
        The reference loss.
        Tensor of shape [1].
    weight : float
        The weight or ratio used for the scaling.
    expected : torch.Tensor
        The expected scaled loss.
        Tensor of shape [1].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scaled = scale_loss(
        loss=loss.to(device), reference=reference.to(device), weight=weight
    )

    assert scaled == expected.to(device)
