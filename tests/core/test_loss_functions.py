import pytest
import torch
from pytest_mock import MockerFixture

from artist.core.loss_functions import (
    AngleLoss,
    FocalSpotLoss,
    KLDivergenceLoss,
    Loss,
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
    base_loss = Loss(loss_function=torch.nn.MSELoss)

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
        (
            torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            None,
            torch.tensor([4.0]),
        ),
    ],
)
def test_vector_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction_dimensions: tuple[int] | None,
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
    reduction_dimensions : tuple[int] | None
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
    if reduction_dimensions is None:
        with pytest.raises(ValueError) as exc_info:
            result = vector_loss(
                prediction=prediction.to(device),
                ground_truth=ground_truth.to(device),
                device=device,
            )
        assert (
            "The vector loss expects reduction_dimensions as keyword argument. Please add this argument."
            in str(exc_info.value)
        )
    else:
        result = vector_loss(
            prediction=prediction.to(device),
            ground_truth=ground_truth.to(device),
            target_area_mask=torch.tensor([0, 1], device=device),
            reduction_dimensions=reduction_dimensions,
            device=device,
        )

        torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, expected, kwargs",
    [
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            torch.tensor([0.0]),
            True,
        ),
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([1.732050776482]),
            True,
        ),
        (
            torch.ones((1, 2, 2)),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([3.0]),
            False,
        ),
    ],
)
def test_focal_spot_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    expected: torch.Tensor,
    kwargs: bool,
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
    kwargs : bool
        Specifies if keyword arguments are passed.
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

    if not kwargs:
        with pytest.raises(ValueError) as exc_info:
            result = focal_spot_loss(
                prediction=prediction.to(device),
                ground_truth=ground_truth.to(device),
            )
        assert (
            "The focal spot loss expects ['reduction_dimensions', 'device', 'target_area_mask'] as keyword arguments. Please add reduction_dimensions as keyword argument. Please add device as keyword argument. Please add target_area_mask as keyword argument."
            in str(exc_info.value)
        )
    else:
        result = focal_spot_loss(
            prediction=prediction.to(device),
            ground_truth=ground_truth.to(device),
            target_area_mask=target_area_mask,
            reduction_dimensions=(1,),
            device=device,
        )

        torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, target_area_dimensions, number_of_rays, expected, kwargs",
    [
        (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[2.0, 2.0]]),
            100,
            torch.tensor([0.0]),
            True,
        ),
        (
            torch.tensor([[[2.0, 3.0], [9.0, 12.0]]]),
            torch.tensor([[[1.0, 2.0], [8.0, 6.0]]]),
            torch.tensor([[2.0, 2.0]]),
            100,
            torch.tensor([39.0]),
            True,
        ),
        (
            torch.tensor([[[2.0, 3.0], [9.0, 12.0]]]),
            torch.tensor([[[1.0, 2.0], [8.0, 6.0]]]),
            torch.tensor([[2.0, 2.0]]),
            100,
            torch.tensor([1.0]),
            False,
        ),
    ],
)
def test_pixel_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    target_area_dimensions: torch.Tensor,
    number_of_rays: int,
    expected: torch.Tensor,
    kwargs: bool,
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
    kwargs : bool
        Specifies if keyword arguments are passed.
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

    if not kwargs:
        with pytest.raises(ValueError) as exc_info:
            result = pixel_loss(
                prediction=prediction.to(device),
                ground_truth=ground_truth.to(device),
            )
        assert (
            "The vector loss expects ['reduction_dimensions', 'device', 'target_area_mask'] as keyword arguments. Please add reduction_dimensions as keyword argument. Please add device as keyword argument. Please add target_area_mask as keyword argument."
            in str(exc_info.value)
        )

    else:
        result = pixel_loss(
            prediction=prediction.to(device),
            ground_truth=ground_truth.to(device),
            target_area_mask=target_area_mask,
            reduction_dimensions=(1, 2),
            device=device,
        )

        torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, expected, kwargs",
    [
        (
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([0.0]),
            True,
        ),
        (
            torch.tensor(
                [
                    [[5.4, 5.71, 2.46], [2.86, 0.44, 0.11]],
                    [[0.5, 0.75, 0.41], [0.11, 2.55, 3.09]],
                ]
            ),
            torch.tensor(
                [
                    [[0.5, 0.75, 0.41], [0.11, 2.55, 3.09]],
                    [[5.4, 5.71, 2.46], [2.86, 0.44, 0.11]],
                ]
            ),
            torch.tensor([2.311237096786, 1.351369142532]),
            True,
        ),
        (
            torch.tensor(
                [
                    [[0.5, 0.75, 0.41], [0.11, 2.55, 3.09]],
                    [[5.4, 5.71, 2.46], [2.86, 0.44, 0.11]],
                ]
            ),
            torch.tensor(
                [
                    [[5.4, 5.71, 2.46], [2.86, 0.44, 0.11]],
                    [[0.5, 0.75, 0.41], [0.11, 2.55, 3.09]],
                ]
            ),
            torch.tensor([1.351369142532, 2.311237096786]),
            True,
        ),
        (
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([0.0]),
            False,
        ),
    ],
)
def test_kl_divergence(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    expected: torch.Tensor,
    kwargs: bool,
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
    kwargs : bool
        Specifies if keyword arguments are passed.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    kl_divergence = KLDivergenceLoss()

    if not kwargs:
        with pytest.raises(ValueError) as exc_info:
            result = kl_divergence(
                prediction=prediction.to(device),
                ground_truth=ground_truth.to(device),
            )
        assert (
            "The kl-divergence loss expects reduction_dimensions as keyword argument. Please add this argument."
            in str(exc_info.value)
        )
    else:
        result = kl_divergence(
            prediction=prediction.to(device),
            ground_truth=ground_truth.to(device),
            target_area_mask=torch.tensor([0, 1], device=device),
            reduction_dimensions=(1, 2),
            device=device,
        )

        torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "prediction, ground_truth, reduction_dimensions, expected",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 3.2, 4.0]]),
            (1),
            torch.tensor([0.0, 7.196009159088e-02]),
        ),
        (
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, -1.0, 0.0]]),
            (1),
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            (1),
            torch.tensor([1.0]),
        ),
    ],
)
def test_angle_loss(
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
        Tensor of variable shape.
    ground_truth : torch.Tensor
        The ground truth.
        Tensor of variable shape.
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
    vector_loss = AngleLoss()
    result = vector_loss(
        prediction=prediction.to(device),
        ground_truth=ground_truth.to(device),
        target_area_mask=torch.tensor([0, 1], device=device),
        reduction_dimensions=reduction_dimensions,
        device=device,
    )

    torch.testing.assert_close(result, expected.to(device), atol=1e-6, rtol=1e-6)
