import pytest
import torch

from artist.core.regularizers import (
    IdealSurfaceRegularizer,
    Regularizer,
    TotalVariationRegularizer,
)


def test_base_regularizer(
    device: torch.device,
) -> None:
    """
    Test the abstract method of the abstract regularizer.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    base_regularizer = Regularizer(weight=1.0, reduction_dimensions=(1,))

    with pytest.raises(NotImplementedError) as exc_info:
        base_regularizer(
            original_surface_points=torch.empty((2, 4), device=device),
            original_surface_normals=torch.empty((2, 4), device=device),
            surface_points=torch.tensor([0, 1], device=device),
            surface_normals=(1,),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)


def test_total_variation_regularizer(
    device: torch.device,
) -> None:
    """
    Test the total variation regularizer.

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

    surface_points = (
        torch.cat([coordinates_smooth, coordinates_irregular], dim=0)
        .unsqueeze(1)
        .expand(2, 4, -1, 3)
    )

    surface_normals = (
        torch.cat([coordinates_smooth, coordinates_irregular * 2], dim=0)
        .unsqueeze(1)
        .expand(2, 4, -1, 3)
    )

    total_variation = TotalVariationRegularizer(
        weight=1.0,
        reduction_dimensions=(1,),
        number_of_neighbors=10,
        sigma=1.0,
    )
    loss_points, loss_normals = total_variation(
        original_surface_points=torch.empty(1, device=device),
        original_surface_normals=torch.empty(1, device=device),
        surface_points=surface_points,
        surface_normals=surface_normals,
        device=device,
    )

    torch.testing.assert_close(
        loss_points,
        torch.tensor([0.174590915442, 2.252339363098], device=device),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        loss_normals,
        torch.tensor([0.174590915442, 4.307547569275], device=device),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "original_surface_points, new_surface_points, original_surface_normals, new_surface_normals, expected_points, expected_normals",
    [
        (
            torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]]]),
            torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]]]),
            torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]),
            torch.tensor([0.3333]),
            torch.tensor([0.5]),
        ),
    ],
)
def test_ideal_surface_regularizer(
    original_surface_points: torch.Tensor,
    new_surface_points: torch.Tensor,
    original_surface_normals: torch.Tensor,
    new_surface_normals: torch.Tensor,
    expected_points: torch.Tensor,
    expected_normals: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the ideal surface regularizer.

    Parameters
    ----------
    original_surface_points : torch.Tensor
        The original surface points.
        Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 3].
    new_surface_points : torch.Tensor
        The new surface points.
        Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 3].
    expected : torch.Tensor
        The expected loss.
        Tensor of shape [number_of_surfaces].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=1.0, reduction_dimensions=(1, 2, 3)
    )
    loss_points, loss_normals = ideal_surface_regularizer(
        original_surface_points=original_surface_points.to(device),
        original_surface_normals=original_surface_normals.to(device),
        surface_points=new_surface_points.to(device),
        surface_normals=new_surface_normals.to(device),
        device=device,
    )

    torch.testing.assert_close(
        loss_points, expected_points.to(device), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        loss_normals, expected_normals.to(device), atol=5e-2, rtol=5e-2
    )
