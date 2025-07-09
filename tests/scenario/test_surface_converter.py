import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.data_loader import paint_loader, stral_loader
from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.scenario.surface_generator import SurfaceGenerator


def test_surface_generator(device: torch.device) -> None:
    """
    Test the surface generator with different conversion methods.

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

    power_plant_position = torch.tensor(
        [50.913421122593, 6.387824755875, 87.000000000000],
        dtype=torch.float64,
        device=device,
    )

    surface_generator_normals = SurfaceGenerator(
        number_of_control_points=torch.tensor([5, 5], device=device),
        step_size=5000,
        conversion_method="deflectometry",
        max_epoch=1,
    )

    surface_generator_points = SurfaceGenerator(
        number_of_control_points=torch.tensor([5, 5], device=device),
        step_size=5000,
        conversion_method="point_cloud",
        max_epoch=1,
    )

    _, facet_translation_vectors, canting, _, _, _ = (
        paint_loader.extract_paint_heliostat_properties(
            heliostat_properties_path=heliostat_file_path,
            power_plant_position=power_plant_position,
            device=device,
        )
    )

    surface_points_with_facets_list, surface_normals_with_facets_list = (
        paint_loader.extract_paint_deflectometry_data(
            heliostat_deflectometry_path=deflectometry_file_path,
            number_of_facets=4,
            device=device,
        )
    )

    surface_config_paint = surface_generator_normals.generate_fitted_surface_config(
        heliostat_name="test",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        device=device,
    )

    (
        facet_translation_vectors,
        canting,
        surface_points_with_facets_list,
        surface_normals_with_facets_list,
    ) = stral_loader.extract_stral_deflectometry_data(
        stral_file_path=stral_file_path, device=device
    )

    surface_config_stral = surface_generator_normals.generate_fitted_surface_config(
        heliostat_name="test",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        device=device,
    )
    surface_config_stral_points = (
        surface_generator_points.generate_fitted_surface_config(
            heliostat_name="test",
            facet_translation_vectors=facet_translation_vectors,
            canting=canting,
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
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


def test_fit_nurbs_conversion_method_error() -> None:
    """
    Test fitting nurbs method for errors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    invalid_method = "invalid-conversion-method"
    with pytest.raises(NotImplementedError) as exc_info:
        SurfaceGenerator(
            number_of_control_points=torch.tensor([5, 5]),
            step_size=5000,
            conversion_method=invalid_method,
            max_epoch=1,
        )
    assert (
        f"The conversion method '{invalid_method}' is not yet supported in ARTIST"
        in str(exc_info.value)
    )
