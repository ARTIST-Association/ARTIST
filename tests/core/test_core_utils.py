import pytest
import torch

from artist.core.core_utils import per_heliostat_reduction, scale_loss


@pytest.mark.parametrize(
    "per_sample_values, active_heliostats_mask, expected",
    [
        (torch.tensor([2.0, 5.0, 6.0, 10.0]), torch.tensor([1, 3]), torch.tensor([2.0, 7.0])),
        (torch.tensor([1.0, 3.0, 5.0, 10.0]), torch.tensor([2, 0, 2]), torch.tensor([2.0, 0.0, 7.5])),
        (torch.tensor([2.0, 4.0]), torch.tensor([2, 0]), torch.tensor([3.0, 0.0,])),
    ]
)
def test_per_heliostat_reduction(
    per_sample_values: torch.Tensor, 
    active_heliostats_mask: torch.Tensor, 
    expected: torch.Tensor, 
    device: torch.device
) -> None:
    """
    Test the per heliostat reduction.

    Parameters
    ----------
    per_sample_values : torch.Tensor
        The loss per sample to be reduced.
        Tensor of shape [number_of_samples].
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
        Tensor of shape [number_of_heliostats].
    expected : torch.Tensor
        The expected reduced loss.
        Tensor of shape [number_of_heliostats].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    result = per_heliostat_reduction(
        per_sample_values=per_sample_values.to(device), 
        active_heliostats_mask=active_heliostats_mask.to(device), 
        device=device
    )

    assert torch.allclose(result, expected.to(device))

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