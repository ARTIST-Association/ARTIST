import pytest
import torch
from pytest_mock import MockerFixture

from artist.core import loss_functions
from artist.field.tower_target_areas import TowerTargetAreas
from artist.scenario.scenario import Scenario


@pytest.mark.parametrize(
    "predictions, targets, reduction_dimensions, expected",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            (1),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            (1),
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            None,
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 5.0], [2.0, 1.0, 3.0, 2.0]]),
            (0, 1),
            torch.tensor([0.3750]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 3.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 5.0], [2.0, 1.0, 3.0, 2.0]]),
            None,
            torch.tensor([0.5]),
        ),
    ],
)
def test_vector_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction_dimensions: tuple[int] | None,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the vector loss function.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted tensor.
        Tensor of shape [number_of_samples, 4].
    targets : torch.Tensor
        The target tensor.
        Tensor of shape [number_of_samples, 4].
    reduction_dimensions : tuple[int] | None
        The dimensions to reduce over.
    expected : torch.Tensor
        The expected loss.
        Tensor of shape [number_of_samples].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    result = loss_functions.vector_loss(
        predictions=predictions.to(device),
        targets=targets.to(device),
        reduction_dimensions=reduction_dimensions,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "predictions, targets, expected",
    [
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            torch.tensor([0.0]),
        ),
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([0.75]),
        ),
    ],
)
def test_focal_spot_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    expected: torch.Tensor,
    mocker: MockerFixture,
    device: torch.device,
) -> None:
    """
    Test the focal spot loss function.

    Parameters
    ----------
    predictions : torch.Tensor
        The flux distributions.
        Tensor of shape [number_of_flux_distributions, e, u].
    targets : torch.Tensor
        The desired focal spots.
        Tensor of shape [number_of_flux_distributions, 4].
    expected : torch.Tensor
        The expected loss.
        Tensor of shape [number_of_flux_distributions].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_scenario = mocker.MagicMock(spec=Scenario)
    target_areas = mocker.MagicMock(spec=TowerTargetAreas)
    target_areas.centers = (torch.tensor([[1.0, 1.0, 1.0, 0.0]], device=device),)
    target_areas.dimensions = (torch.tensor([[2, 2]], device=device),)
    mock_scenario.target_areas = target_areas

    target_area_mask = torch.tensor([0], device=device)

    result = loss_functions.focal_spot_loss(
        predictions=predictions.to(device),
        targets=targets.to(device),
        scenario=mock_scenario,
        target_area_mask=target_area_mask,
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "predictions, targets, target_area_dimensions, number_of_rays, expected",
    [
        (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[2.0, 2.0]]),
            100,
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[[2.0, 3.0], [9.0, 12.0]]]),
            torch.tensor([[[1.0, 2.0], [8.0, 6.0]]]),
            torch.tensor([[2.0, 2.0]]),
            100,
            torch.tensor([0.190476119518]),
        ),
    ],
)
def test_pixel_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_area_dimensions: torch.Tensor,
    number_of_rays: int,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the pixel loss function.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted flux distribution.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target flux distribution.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    target_area_dimensions : torch.Tensor
        The dimensions of the tower target areas aimed at.
        Tensor of shape [number_of_flux_distributions, 2].
    number_of_rays : int
        The number of rays used to generate the flux.
    expected : torch.Tensor
        The expected pixel loss.
        Tensor of shape [number_of_flux_distributions].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    result = loss_functions.pixel_loss(
        predictions=predictions.to(device),
        targets=targets.to(device),
        target_area_dimensions=target_area_dimensions.to(device),
        number_of_rays=number_of_rays,
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "predictions, targets, expected",
    [
        (
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([0.693147182465]),
        ),
        (
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([13.122363090515]),
        ),
    ],
)
def test_kl_divergence(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the Kullback-Leibler divergence function.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    expected : torch.Tensor
        The expected KL divergence result.
        Tensor of shape [number_of_flux_distributions].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    result = loss_functions.kl_divergence(
        predictions=predictions.to(device), targets=targets.to(device)
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "loss, reference_loss, weight, expected",
    [
        (torch.tensor([2.0]), torch.tensor([2.0]), 1.0, torch.tensor([2.0])),
        (torch.tensor([2.0]), torch.tensor([2.0]), 0.5, torch.tensor([1.0])),
        (torch.tensor([4.0]), torch.tensor([2.0]), 1.0, torch.tensor([2.0])),
        (torch.tensor([0.0]), torch.tensor([2.0]), 1.0, torch.tensor([0.0])),
    ],
)
def test_scale_loss(
    loss: torch.Tensor,
    reference_loss: torch.Tensor,
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
    reference_loss :  torch.Tensor
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
    scaled = loss_functions.scale_loss(
        loss=loss.to(device), reference_loss=reference_loss.to(device), weight=weight
    )

    assert scaled == expected.to(device)


def test_total_variation_loss(
    device: torch.device,
) -> None:
    """
    Test the total variation loss function.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    rows = torch.linspace(0, 4 * 3.1415, 120, device=device)
    columns = torch.linspace(0, 4 * 3.1415, 120, device=device)
    x, y = torch.meshgrid(rows, columns, indexing="ij")

    # Smooth surface with waves.
    z_values_smooth = 0.5 * torch.sin(x) + 0.5 * torch.cos(y)

    # Irregular surface = smooth surface with waves and random noise.
    noise = torch.randn_like(z_values_smooth, device=device) * 0.5
    z_irregular = z_values_smooth + noise

    coordinates_smooth = torch.stack(
        [x.flatten(), y.flatten(), z_values_smooth.flatten()], dim=1
    ).unsqueeze(0)
    coordinates_irregular = torch.stack(
        [x.flatten(), y.flatten(), z_irregular.flatten()], dim=1
    ).unsqueeze(0)

    surfaces = (
        torch.cat([coordinates_smooth, coordinates_irregular], dim=0)
        .unsqueeze(1)
        .expand(2, 4, -1, 3)
    )

    # Calculate total variation loss
    loss = loss_functions.total_variation_loss(
        surfaces=surfaces, number_of_neighbors=10, sigma=1.0, device=device
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(
            [
                [0.043646000326, 0.043646000326, 0.043646000326, 0.043646000326],
                [0.564661264420, 0.564661264420, 0.564661264420, 0.564661264420],
            ],
            device=device,
        ),
        atol=5e-2,
        rtol=5e-2,
    )
