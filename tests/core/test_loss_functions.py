import pytest
import torch
from pytest_mock import MockerFixture

from artist.core.loss_functions import (
    BaseLoss,
    FocalSpotLoss,
    KLDivergenceLoss,
    PixelLoss,
    VectorLoss,
)
from artist.field.tower_target_areas import TowerTargetAreas
from artist.scenario.scenario import Scenario
from artist.scene.light_source import LightSource
from artist.scene.light_source_array import LightSourceArray


def test_base_loss(
    device: torch.device,
) -> None:
    """
    Test the abstract method of the abstract base loss.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    base_loss = BaseLoss()
    with pytest.raises(NotImplementedError) as exc_info:
        base_loss(
            prediction=torch.empty((2, 4), device=device),
            ground_truth=torch.empty((2, 4), device=device),
            target_area_mask=torch.tensor([0, 1], device=device),
            reduction_dimensions=(1,),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)


@pytest.mark.parametrize(
    "prediction, ground_truth, reduction_dimensions, expected",
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
            torch.tensor([4.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 5.0], [2.0, 1.0, 3.0, 2.0]]),
            (1),
            torch.tensor([1.0, 2.0]),
        ),
    ],
)
def test_vector_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction_dimensions: tuple[int],
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the vector loss.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted values.
        Tensor of shape [number_of_samples, 4].
    ground_truth : torch.Tensor
        The ground truth.
        Tensor of shape [number_of_samples, 4].
    reduction_dimensions : tuple[int]
        The dimensions along which to reduce the final loss.
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
    vector_loss = VectorLoss()
    result = vector_loss(
        prediction=prediction.to(device),
        ground_truth=ground_truth.to(device),
        target_area_mask=torch.tensor([0, 1], device=device),
        reduction_dimensions=reduction_dimensions,
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, expected",
    [
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            torch.tensor([0.0]),
        ),
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([3.0]),
        ),
    ],
)
def test_focal_spot_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    expected: torch.Tensor,
    mocker: MockerFixture,
    device: torch.device,
) -> None:
    """
    Test the focal spot loss.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted values.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    ground_truth : torch.Tensor
        The ground truth.
        Tensor of shape [number_of_samples, 4].
    expected : torch.Tensor
        The expected loss.
        Tensor of shape [number_of_flux_distributions].
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
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

    focal_spot_loss = FocalSpotLoss(scenario=mock_scenario)
    result = focal_spot_loss(
        prediction=prediction.to(device),
        ground_truth=ground_truth.to(device),
        target_area_mask=target_area_mask,
        reduction_dimensions=(1,),
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, target_area_dimensions, number_of_rays, expected",
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
            torch.tensor([0.761904418468]),
        ),
    ],
)
def test_pixel_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    target_area_dimensions: torch.Tensor,
    number_of_rays: int,
    expected: torch.Tensor,
    mocker: MockerFixture,
    device: torch.device,
) -> None:
    """
    Test the pixel loss.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted values.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    ground_truth : torch.Tensor
        The ground truth.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    target_area_dimensions : torch.Tensor
        The dimensions of the tower target areas aimed at.
        Tensor of shape [number_of_samples, 2].
    number_of_rays : int
        The number of rays used to generate the flux.
    expected : torch.Tensor
        The expected pixel loss.
        Tensor of shape [number_of_samples].
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_scenario = mocker.MagicMock(spec=Scenario)

    target_areas = mocker.MagicMock(spec=TowerTargetAreas)
    target_areas.dimensions = target_area_dimensions.to(device)
    mock_scenario.target_areas = target_areas

    light_sources = mocker.MagicMock(spec=LightSourceArray)
    light_source = mocker.MagicMock(spec=LightSource)
    light_source.number_of_rays = number_of_rays
    light_sources.light_source_list = [light_source]
    mock_scenario.light_sources = light_sources

    target_area_mask = torch.tensor([0], device=device)

    pixel_loss = PixelLoss(scenario=mock_scenario)
    result = pixel_loss(
        prediction=prediction.to(device),
        ground_truth=ground_truth.to(device),
        target_area_mask=target_area_mask,
        reduction_dimensions=(1, 2),
        number_of_rays=number_of_rays,
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, expected",
    [
        (
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[[0.5, 0.75]]]),
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([27.631021499634]),
        ),
        (
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[0.5, 0.75]]]),
            torch.tensor([0.916290760040]),
        ),
    ],
)
def test_kl_divergence(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the Kullback-Leibler divergence function.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted values.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    ground_truth : torch.Tensor
        The ground truth.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
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
    kl_divergence = KLDivergenceLoss()
    result = kl_divergence(
        prediction=prediction.to(device),
        ground_truth=ground_truth.to(device),
        target_area_mask=torch.tensor([0, 1], device=device),
        reduction_dimensions=(1, 2),
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)
