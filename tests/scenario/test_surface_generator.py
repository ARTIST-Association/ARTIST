import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.data_parser import paint_scenario_parser, stral_scenario_parser
from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import config_dictionary


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
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

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

    surface_generator = SurfaceGenerator(
        number_of_control_points=torch.tensor([5, 5], device=device), device=device
    )

    _, facet_translation_vectors, canting, _, _, _ = (
        paint_scenario_parser.extract_paint_heliostat_properties(
            heliostat_properties_path=heliostat_file_path,
            power_plant_position=power_plant_position,
            device=device,
        )
    )

    surface_points_with_facets_list, surface_normals_with_facets_list = (
        paint_scenario_parser.extract_paint_deflectometry_data(
            heliostat_deflectometry_path=deflectometry_file_path,
            number_of_facets=4,
            device=device,
        )
    )

    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    surface_config_paint_fit_normals = surface_generator.generate_fitted_surface_config(
        heliostat_name="test",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        optimizer=optimizer,
        scheduler=scheduler,
        deflectometry_step_size=5000,
        fit_method=config_dictionary.fit_nurbs_from_normals,
        max_epoch=1,
        device=device,
    )

    (
        facet_translation_vectors,
        canting,
        surface_points_with_facets_list,
        surface_normals_with_facets_list,
    ) = stral_scenario_parser.extract_stral_deflectometry_data(
        stral_file_path=stral_file_path, device=device
    )

    surface_config_stral_fit_normals = surface_generator.generate_fitted_surface_config(
        heliostat_name="test",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        optimizer=optimizer,
        scheduler=scheduler,
        deflectometry_step_size=5000,
        fit_method=config_dictionary.fit_nurbs_from_normals,
        max_epoch=1,
        device=device,
    )
    surface_config_stral_fit_points = surface_generator.generate_fitted_surface_config(
        heliostat_name="test",
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        optimizer=optimizer,
        scheduler=scheduler,
        deflectometry_step_size=5000,
        fit_method=config_dictionary.fit_nurbs_from_points,
        max_epoch=1,
        device=device,
    )

    surface_config_ideal = surface_generator.generate_ideal_surface_config(
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        device=device,
    )

    assert isinstance(surface_config_paint_fit_normals, SurfaceConfig)
    assert isinstance(surface_config_stral_fit_normals, SurfaceConfig)
    assert isinstance(surface_config_stral_fit_points, SurfaceConfig)
    assert isinstance(surface_config_ideal, SurfaceConfig)
    assert all(
        isinstance(obj, FacetConfig)
        for obj in surface_config_paint_fit_normals.facet_list
    )
    assert all(
        isinstance(obj, FacetConfig)
        for obj in surface_config_stral_fit_normals.facet_list
    )
    assert all(
        isinstance(obj, FacetConfig)
        for obj in surface_config_stral_fit_points.facet_list
    )
    assert all(isinstance(obj, FacetConfig) for obj in surface_config_ideal.facet_list)

    torch.testing.assert_close(
        surface_config_paint_fit_normals.facet_list[1].control_points[0, 3],
        torch.tensor([0.015739023685, 0.943195939064, 0.038207393140], device=device),
    )
    torch.testing.assert_close(
        surface_config_stral_fit_normals.facet_list[0].control_points[4, 2],
        torch.tensor([-0.015720605850, 0.643227100372, 0.038244392723], device=device),
    )
    torch.testing.assert_close(
        surface_config_stral_fit_points.facet_list[3].control_points[0, 0],
        torch.tensor([-0.790499985218, -0.601999938488, 0.001999197528], device=device),
    )
    torch.testing.assert_close(
        surface_config_ideal.facet_list[2].control_points[3, 2],
        torch.tensor([0.401250004768, 0.000000000000, 0.000000000000], device=device),
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
        surface_generator = SurfaceGenerator()

        optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=50,
            threshold=1e-7,
            threshold_mode="abs",
        )

        surface_generator.generate_fitted_surface_config(
            heliostat_name="test",
            facet_translation_vectors=torch.empty((1, 3), device=device),
            canting=torch.empty((1, 2, 3), device=device),
            surface_points_with_facets_list=[torch.empty((1, 3), device=device)],
            surface_normals_with_facets_list=[torch.empty((1, 3), device=device)],
            optimizer=optimizer,
            scheduler=scheduler,
            fit_method=invalid_method,
            device=device,
        )

    assert (
        f"The conversion method '{invalid_method}' is not yet supported in ARTIST"
        in str(exc_info.value)
    )
