import pytest
import torch

from artist.core.regularizers import (
    IdealSurfaceRegularizer,
    Regularizer,
    SmoothnessRegularizer,
)

torch.manual_seed(7)
torch.cuda.manual_seed(7)


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
            current_control_points=torch.empty((1, 1, 6, 6, 4), device=device),
            original_control_points=torch.empty((1, 1, 6, 6, 4), device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)


@pytest.fixture
def control_points(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate flat, smooth, and irregular control point tensors.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        Flat control points.
        Tensor of shape [number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
    torch.Tensor
        Smooth control points.
        Tensor of shape [number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
    torch.Tensor
        Irregular control points.
        Tensor of shape [number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
    """
    x = torch.linspace(0, 4 * 3.1415, 6, device=device)
    y = torch.linspace(0, 4 * 3.1415, 6, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    x_expanded = x_grid.unsqueeze(0).expand(4, -1, -1)
    y_expanded = y_grid.unsqueeze(0).expand(4, -1, -1)

    # Flat surface
    z_flat = torch.zeros_like(x_expanded, device=device)
    flat_points = torch.stack([x_expanded, y_expanded, z_flat], dim=-1)

    # Smooth surface
    z_smooth = 0.2 * torch.sin(x_expanded) + 0.2 * torch.cos(y_expanded)
    smooth_points = torch.stack([x_expanded, y_expanded, z_smooth], dim=-1)

    # Irregular surface
    z_irregular = z_smooth * 5.0
    irregular_points = torch.stack([x_expanded, y_expanded, z_irregular], dim=-1)

    return flat_points, smooth_points, irregular_points


def test_smoothness_regularizer(
    control_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the smoothness regularizer.

    Parameters
    ----------
    control_points : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A fixture to retrieve the control points.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    flat_points, smooth_points, irregular_points = control_points

    smoothness_regularizer = SmoothnessRegularizer(
        weight=1.0,
        reduction_dimensions=(1,),
    )
    loss = smoothness_regularizer(
        current_control_points=torch.stack([smooth_points, irregular_points]),
        original_control_points=flat_points.expand(2, 4, 6, 6, 3),
    )

    torch.testing.assert_close(
        loss,
        torch.tensor([0.529724955559, 13.243123054504], device=device),
        atol=5e-4,
        rtol=5e-4,
    )


def test_ideal_surface_regularizer(control_points, device):
    """
    Test the ideal surface regularizer.

    Parameters
    ----------
    control_points : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A fixture to retrieve the control points.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    flat_points, smooth_points, irregular_points = control_points

    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=1.0,
        reduction_dimensions=(1,),
    )
    loss = ideal_surface_regularizer(
        current_control_points=torch.stack([smooth_points, irregular_points]),
        original_control_points=flat_points.expand(2, 4, 6, 6, 3),
    )

    torch.testing.assert_close(
        loss,
        torch.tensor([0.053332783282, 1.333319664001], device=device),
        atol=5e-4,
        rtol=5e-4,
    )
