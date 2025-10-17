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

    surfaces = (
        torch.cat([coordinates_smooth, coordinates_irregular], dim=0)
        .unsqueeze(1)
        .expand(2, 4, -1, 3)
    )

    total_variation = TotalVariationRegularizer(
        weight=1.0,
        reduction_dimensions=(1,),
        surface="surface_points",
        number_of_neighbors=10,
        sigma=1.0,
    )
    loss = total_variation(
        original_surface_points=torch.empty(1, device=device),
        surface_points=surfaces,
        surface_normals=torch.empty(1, device=device),
        device=device,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor([0.174590915442, 2.252339363098], device=device),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "original_surface_points, new_surface_points, expected",
    [
        (
            torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]]]),
            torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]]]),
            torch.tensor([2.0]),
        ),
    ],
)
def test_ideal_surface_regularizer(
    original_surface_points: torch.Tensor,
    new_surface_points: torch.Tensor,
    expected: torch.Tensor,
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
    loss = ideal_surface_regularizer(
        original_surface_points=original_surface_points.to(device),
        surface_points=new_surface_points.to(device),
        surface_normals=torch.empty(1, device=device),
        device=device,
    )

    torch.testing.assert_close(loss, expected.to(device), atol=5e-2, rtol=5e-2)
