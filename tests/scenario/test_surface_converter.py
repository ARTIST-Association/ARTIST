import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.scenario.surface_converter import SurfaceConverter


def test_surface_converter(device: torch.device) -> None:
    """
    Test the surface converter with ``STRAL`` and ``PAINT`` files.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    heliostat_file_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/field_data/AA39-heliostat-properties.json"
    )
    deflectometry_file_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA39-deflectometry.h5"
    )
    stral_file_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/test_stral_data.binp"
    )

    surface_converter_normals = SurfaceConverter(
        step_size=5000,
        number_of_evaluation_points=torch.tensor([10, 10]),
        conversion_method="deflectometry",
        number_control_points=torch.tensor([5, 5]),
        max_epoch=1,
    )

    surface_converter_points = SurfaceConverter(
        step_size=5000,
        number_of_evaluation_points=torch.tensor([10, 10]),
        conversion_method="point_cloud",
        number_control_points=torch.tensor([5, 5]),
        max_epoch=1,
    )

    surface_config_paint = surface_converter_normals.generate_surface_config_from_paint(
        heliostat_file_path=heliostat_file_path,
        deflectometry_file_path=deflectometry_file_path,
        device=device,
    )

    surface_config_stral = surface_converter_normals.generate_surface_config_from_stral(
        stral_file_path=stral_file_path,
        device=device,
    )

    surface_config_stral_points = (
        surface_converter_points.generate_surface_config_from_stral(
            stral_file_path=stral_file_path,
            device=device,
        )
    )

    assert isinstance(surface_config_paint, SurfaceConfig)
    assert isinstance(surface_config_stral, SurfaceConfig)
    assert isinstance(surface_config_stral_points, SurfaceConfig)
    assert all(isinstance(obj, FacetConfig) for obj in surface_config_paint.facet_list)
    assert all(isinstance(obj, FacetConfig) for obj in surface_config_stral.facet_list)
    assert all(
        isinstance(obj, FacetConfig) for obj in surface_config_stral_points.facet_list
    )


def test_fit_nurbs_conversion_method_error(device: torch.device) -> None:
    """
    Test fitting nurbs method for errors.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    invalid_method = "invalid-conversion-method"
    with pytest.raises(NotImplementedError) as exc_info:
        SurfaceConverter(
            step_size=5000,
            number_of_evaluation_points=torch.tensor([10, 10]),
            number_control_points=torch.tensor([5, 5]),
            conversion_method=invalid_method,
            max_epoch=1,
        )
    assert (
        f"The conversion method '{invalid_method}' is not yet supported in ARTIST"
        in str(exc_info.value)
    )
