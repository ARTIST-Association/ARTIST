import argparse
import csv
import json
import pathlib
import re
import warnings
from typing import Any, cast

import h5py
import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_reconstructor import KinematicReconstructor
from artist.core.loss_functions import FocalSpotLoss, KLDivergenceLoss
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.core.regularizers import IdealSurfaceRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario import Scenario
from artist.util import (
    config_dictionary,
    index_mapping,
    runtime_log,
    set_logger_config,
    track_runtime,
    utils,
)
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces

set_logger_config()
torch.manual_seed(7)
torch.cuda.manual_seed(7)


def get_incremented_path_number(base_path: pathlib.Path) -> int:
    """
    Store the results of each run incrementally, this function increases the number.

    Parameters
    ----------
    base_path : pathlib.Path
        The base path where the results are saved.

    Returns
    -------
    int
        The number of the next results file.
    """
    stem = base_path.stem
    suffix = base_path.suffix

    pattern = re.compile(rf"^{re.escape(stem)}(?:_(\d+))?{re.escape(suffix)}$")
    existing_numbers = []

    for file in base_path.parent.glob(f"{stem}*{suffix}"):
        match = pattern.match(file.name)
        if match:
            number = match.group(1)
            if number is None:
                existing_numbers.append(0)
            else:
                existing_numbers.append(int(number))

    next_number = 0
    while next_number in existing_numbers:
        next_number += 1

    runtime_log.info(f"Results_file_number: {next_number}")

    return next_number


@track_runtime(runtime_log)
def create_distributions(
    measured_data_dir: pathlib.Path,
    results_dir: pathlib.Path,
    results_number: int,
    device: torch.device | None = None,
) -> None:
    """
    Save the baseline measured distribution and a homogeneous distribution in the results file.

    Parameters
    ----------
    measured_data_dir : pathlib.Path
        Path to the measured baseline data.
    results_dir : pathlib.Path
        Path to where the results are saved.
    results_number : int
        The current, incremented results number.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device=device)

    results_dict: dict[str, dict[str, torch.Tensor]] = {}
    results_path = results_dir / f"results_{results_number}.pt"

    try:
        loaded = torch.load(results_path, weights_only=False)
        results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)
    except FileNotFoundError:
        print(f"File not found: {results_path}. Initializing with an empty dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

    if "measured_flux" not in results_dict.keys():
        measured_flux_path_csv = measured_data_dir / "measured_flux.csv"
        data = []

        with open(measured_flux_path_csv, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                data.append([float(x) for x in row])

        bitmap_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        bitmap_resized = bitmap_tensor.unsqueeze(0).unsqueeze(0)
        measured_flux = torch.nn.functional.interpolate(
            bitmap_resized, size=(256, 256), mode="bilinear", align_corners=True
        ).squeeze()

        results_dict["measured_flux"] = measured_flux

    if "homogeneous_distribution" not in results_dict.keys():
        e_trapezoid = utils.trapezoid_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        u_trapezoid = utils.trapezoid_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        eu_trapezoid = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

        homogeneous_distribution = eu_trapezoid / eu_trapezoid.sum()

        results_dict["homogeneous_distribution"] = homogeneous_distribution

    torch.save(results_dict, results_path)


@track_runtime(runtime_log)
def create_deflectometry_surface_for_comparison(
    scenario_path: pathlib.Path,
    results_dir: pathlib.Path,
    results_number: int,
    measured_data_dir: pathlib.Path,
    heliostats_for_plots: list[str],
    reconstruction_parameters: dict[str, float | int],
    device: torch.device | None,
) -> None:
    """
    Create the surface from the measured deflectometry as comparison.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    results_dir : pathlib.Path
        Path to where the results are saved.
    results_number : int
        The current, incremented results number.
    measured_data_dir : pathlib.Path
        Path to the measured deflectometry data.
    heliostats_for_plots : list[str]
        The selected heliostat names used for the evaluation plots.
    reconstruction_parameters : dict[str, float | int]
        Parameters for the reconstruction.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device)

    results_path = results_dir / f"results_{results_number}.pt"
    results_dict: dict[str, dict[str, torch.Tensor]] = {}

    loaded = torch.load(results_path, weights_only=False)
    results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        number_of_surface_points_per_facet = torch.tensor(
            [
                reconstruction_parameters["number_of_surface_points"],
                reconstruction_parameters["number_of_surface_points"],
            ],
            device=device,
        )

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                device=device,
            )

        fitted_deflec_data = {}

        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            for i, heliostat in enumerate(heliostat_group.names):
                fitted_deflec_data[heliostat] = torch.stack(
                    (
                        heliostat_group.surface_points[i],
                        heliostat_group.surface_normals[i],
                    )
                )

    results_dict["deflectometry_fitted"] = fitted_deflec_data

    original_deflec_data = {}
    for name in heliostats_for_plots:
        path = measured_data_dir / f"deflec_{name}.pt"
        data = torch.load(path, weights_only=False)
        original_deflec_data[name] = data

    results_dict["deflectometry_original"] = original_deflec_data

    torch.save(results_dict, results_path)


def save_heliostat_model(
    scenario: Scenario, save_dir: pathlib.Path, results_number: int
) -> None:
    """
    Save the current heliostat model.

    Parameters
    ----------
    scenario : Scenario
        The scenario.
    save_dir : pathlib.Path
        Directory path where the data will be saved.
    results_number : int
        The incremented number of the results file.
    """
    data: dict[str, list] = {
        "names": [],
        "positions": [],
        "widths": [],
        "heights": [],
        "number_of_facets": [],
        "axis_offsets": [],
        "mirror_offsets": [],
        "facet_translations": [],
        "canting_vectors": [],
        "surface_points": [],
        "surface_normals": [],
    }

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        data["names"].extend(heliostat_group.names)
        data["positions"].extend(
            tuple(position[:3].tolist()) for position in heliostat_group.positions
        )
        start = -torch.norm(heliostat_group.canting, dim=index_mapping.canting)
        end = torch.norm(heliostat_group.canting, dim=index_mapping.canting)
        data["widths"].extend(((end[:, 0] - start[:, 0]) * 2)[:, 0] + 0.01)
        data["heights"].extend(((end[:, 0] - start[:, 0]) * 2)[:, 1] + 0.01)
        data["number_of_facets"].extend(
            [(heliostat_group.number_of_facets_per_heliostat, 1)]
            * heliostat_group.number_of_heliostats
        )
        data["axis_offsets"].extend([0.0] * heliostat_group.number_of_heliostats)
        data["mirror_offsets"].extend([0.0] * heliostat_group.number_of_heliostats)
        data["facet_translations"].extend(heliostat_group.facet_translations)
        data["canting_vectors"].extend(heliostat_group.canting)
        data["surface_points"].extend(
            heliostat_group.surface_points.reshape(
                heliostat_group.number_of_heliostats,
                heliostat_group.number_of_facets_per_heliostat,
                -1,
                4,
            )
        )
        data["surface_normals"].extend(
            heliostat_group.surface_normals.reshape(
                heliostat_group.number_of_heliostats,
                heliostat_group.number_of_facets_per_heliostat,
                -1,
                4,
            )
        )

    torch.save(data, save_dir / f"reconstructed_heliostats_data_{results_number}.pt")


@track_runtime(runtime_log)
def align_and_trace_rays(
    scenario: Scenario,
    ddp_setup: dict[str, Any],
    incident_ray_direction: torch.Tensor,
    target_area_index: int,
    aim_point: torch.Tensor,
    motor_positions: torch.Tensor | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Align heliostats and trace rays to create a flux density prediction on the target.

    Parameters
    ----------
    scenario : Scenario
        The scenario.
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    incident_ray_direction : torch.Tensor
        The incident ray direction.
    target_area_index : int
        The target area index.
    aim_point : torch.Tensor
        Aim point of the baseline measurement.
    motor_positions : torch.Tensor
        The optimized motor positions, if None the heliostats will be aligned by the incident ray direction (default is None).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Flux density distribution on the target.
    """
    device = get_device(device)

    bitmap_resolution = torch.tensor([256, 256])

    combined_bitmaps_per_target = torch.zeros(
        (
            scenario.target_areas.number_of_target_areas,
            bitmap_resolution[index_mapping.unbatched_bitmap_e],
            bitmap_resolution[index_mapping.unbatched_bitmap_u],
        ),
        device=device,
    )

    for heliostat_group_index in ddp_setup[config_dictionary.groups_to_ranks_mapping][
        ddp_setup[config_dictionary.rank]
    ]:
        heliostat_group: HeliostatGroup = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]

        (
            active_heliostats_mask,
            target_area_mask,
            incident_ray_directions,
        ) = scenario.index_mapping(
            heliostat_group=heliostat_group,
            single_incident_ray_direction=incident_ray_direction,
            single_target_area_index=target_area_index,
            device=device,
        )

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask, device=device
        )

        if motor_positions is None:
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=aim_point.expand(
                    heliostat_group.number_of_active_heliostats, 4
                ),
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

        else:
            heliostat_group.align_surfaces_with_motor_positions(
                motor_positions=motor_positions[heliostat_group_index],
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

        scenario.set_number_of_rays(number_of_rays=3)

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            world_size=ddp_setup[config_dictionary.heliostat_group_world_size],
            rank=ddp_setup[config_dictionary.heliostat_group_rank],
            batch_size=100,
            random_seed=ddp_setup[config_dictionary.heliostat_group_rank],
            bitmap_resolution=bitmap_resolution,
        )

        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
            bitmaps_per_heliostat=bitmaps_per_heliostat,
            target_area_mask=target_area_mask,
            device=device,
        )

        combined_bitmaps_per_target = combined_bitmaps_per_target + bitmaps_per_target

    if ddp_setup[config_dictionary.is_nested]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target,
            op=torch.distributed.ReduceOp.SUM,
            group=ddp_setup[config_dictionary.process_subgroup],
        )

    if ddp_setup[config_dictionary.is_distributed]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
        )

    return combined_bitmaps_per_target


def merge_data(
    unoptimized_data: dict[str, dict[str, torch.Tensor]],
    optimized_data: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Merge data dictionaries.

    Parameters
    ----------
    unoptimized_data : dict[str, dict[str, torch.Tensor]]
        Data dictionary containing unoptimized data.
    optimized_data : dict[str, dict[str, torch.Tensor]]
        Data dictionary containing optimized data.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        The combined data dictionary.
    """
    merged = {}

    for heliostat in unoptimized_data.keys():
        fluxes = torch.stack(
            (
                unoptimized_data[heliostat]["measured_flux"],
                unoptimized_data[heliostat]["artist_flux"],
                optimized_data[heliostat]["artist_flux"],
            )
        )

        merged[heliostat] = {
            "fluxes": fluxes,
        }

        if len(unoptimized_data[heliostat]) > 2:
            surface_points = torch.stack(
                (
                    unoptimized_data[heliostat]["surface_points"],
                    optimized_data[heliostat]["surface_points"],
                )
            )
            surface_normals = torch.stack(
                (
                    unoptimized_data[heliostat]["surface_normals"],
                    optimized_data[heliostat]["surface_normals"],
                )
            )
            canting = optimized_data[heliostat]["canting"]
            facet_translations = optimized_data[heliostat]["facet_translations"]

            merged[heliostat] = {
                "fluxes": fluxes,
                "surface_points": surface_points,
                "surface_normals": surface_normals,
                "canting": canting,
                "facet_translations": facet_translations,
            }

    return merged


def kinematic_data(
    scenario: Scenario,
    ddp_setup: dict[str, Any],
    heliostat_data: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Extract heliostat kinematic information.

    Parameters
    ----------
    scenario : Scenario
        The scenario.
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    heliostat_data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        Heliostat and calibration measurement data.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Dictionary containing kinematic data per heliostat.
    """
    device = get_device(device)

    bitmaps_for_plots = {}

    for heliostat_group_index in ddp_setup[config_dictionary.groups_to_ranks_mapping][
        ddp_setup[config_dictionary.rank]
    ]:
        heliostat_group: HeliostatGroup = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]

        parser = cast(
            CalibrationDataParser, heliostat_data[config_dictionary.data_parser]
        )
        heliostat_mapping = cast(
            list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
            heliostat_data[config_dictionary.heliostat_data_mapping],
        )
        (
            measured_fluxes,
            _,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask, device=device
        )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        scenario.set_number_of_rays(number_of_rays=100)

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            world_size=ddp_setup[config_dictionary.heliostat_group_world_size],
            rank=ddp_setup[config_dictionary.heliostat_group_rank],
            batch_size=heliostat_group.number_of_active_heliostats,
            random_seed=ddp_setup[config_dictionary.heliostat_group_rank],
        )

        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        names = [
            heliostat_group.names[i]
            for i in torch.nonzero(active_heliostats_mask).squeeze()
        ]

        for i, heliostat in enumerate(names):
            bitmaps_for_plots[heliostat] = {
                "artist_flux": bitmaps_per_heliostat[i],
                "measured_flux": measured_fluxes[i],
            }

    return bitmaps_for_plots


def surface_data(
    scenario: Scenario,
    ddp_setup: dict[str, Any],
    heliostat_data: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ],
    number_of_surface_points_per_facet: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Extract heliostat surface information.

    Parameters
    ----------
    scenario : Scenario
        The scenario.
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    heliostat_data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        Heliostat and calibration measurement data.
    number_of_surface_points_per_facet : torch.Tensor
        Number of surface points per facet in east and north direction.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Dictionary containing surface data per heliostat.
    """
    device = get_device(device=device)

    data_for_plots = {}

    for heliostat_group_index in ddp_setup[config_dictionary.groups_to_ranks_mapping][
        ddp_setup[config_dictionary.rank]
    ]:
        heliostat_group: HeliostatGroup = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]

        parser = cast(
            CalibrationDataParser, heliostat_data[config_dictionary.data_parser]
        )
        heliostat_mapping = cast(
            list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
            heliostat_data[config_dictionary.heliostat_data_mapping],
        )
        (
            measured_fluxes,
            _,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask, device=device
        )

        evaluation_points = (
            utils.create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_surface_points_per_facet,
                device=device,
            )
            .unsqueeze(index_mapping.heliostat_dimension)
            .unsqueeze(index_mapping.facet_index_unbatched)
            .expand(
                heliostat_group.number_of_active_heliostats,
                heliostat_group.number_of_facets_per_heliostat,
                -1,
                -1,
            )
        )
        nurbs_surfaces = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.nurbs_control_points[
                active_heliostats_mask == 1
            ],
            device=device,
        )
        (
            surface_points,
            surface_normals,
        ) = nurbs_surfaces.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points,
            canting=heliostat_group.canting[active_heliostats_mask == 1],
            facet_translations=heliostat_group.facet_translations[
                active_heliostats_mask == 1
            ],
            device=device,
        )

        heliostat_group.active_surface_points = surface_points.reshape(
            active_heliostats_mask.sum(), -1, 4
        )
        heliostat_group.active_surface_normals = surface_normals.reshape(
            active_heliostats_mask.sum(), -1, 4
        )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        scenario.set_number_of_rays(number_of_rays=100)

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            world_size=ddp_setup[config_dictionary.heliostat_group_world_size],
            rank=ddp_setup[config_dictionary.heliostat_group_rank],
            batch_size=heliostat_group.number_of_active_heliostats,
            random_seed=ddp_setup[config_dictionary.heliostat_group_rank],
        )

        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        cropped_flux_distributions = utils.crop_flux_distributions_around_center(
            flux_distributions=bitmaps_per_heliostat,
            crop_width=config_dictionary.utis_crop_width,
            crop_height=config_dictionary.utis_crop_height,
            target_plane_widths=scenario.target_areas.dimensions[target_area_mask][
                :, index_mapping.target_area_width
            ],
            target_plane_heights=scenario.target_areas.dimensions[target_area_mask][
                :, index_mapping.target_area_height
            ],
            device=device,
        )

        names = [
            heliostat_group.names[i]
            for i in torch.nonzero(active_heliostats_mask).squeeze()
        ]

        for index, heliostat in enumerate(names):
            data_for_plots[heliostat] = {
                "measured_flux": measured_fluxes[index],
                "artist_flux": cropped_flux_distributions[index],
                "surface_points": surface_points[index],
                "surface_normals": surface_normals[index],
                "canting": heliostat_group.active_canting[index],
                "facet_translations": heliostat_group.active_facet_translations[index],
            }

    return data_for_plots


def create_surface_reconstruction_batches(
    heliostat_data: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: CalibrationDataParser,
    batch_size: int,
) -> list[
    dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ]
]:
    """
    Create batches of data for the surface reconstruction to avoid out of memory errors.

    Parameters
    ----------
    heliostat_data : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mappings including heliostat names and their calibration files, for surface reconstruction.
    data_parser : CalibrationDataParser
        Data parser for the calibration data files.
    batch_size : int
        Number of measurements in one batch.

    Returns
    -------
    list[dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]]:
        Batches of surfaces reconstruction data in a list.
    """
    if heliostat_data is None:
        return []

    data_surfaces: list[
        dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ]
    ] = []

    for i in range(0, len(heliostat_data), batch_size):
        batch = heliostat_data[i : i + batch_size]
        data_surfaces.append(
            {
                config_dictionary.data_parser: data_parser,
                config_dictionary.heliostat_data_mapping: batch,
            }
        )

    return data_surfaces


@track_runtime(runtime_log)
def ablation_study(
    scenario_path: pathlib.Path,
    results_dir: pathlib.Path,
    results_number: int,
    baseline_incident_ray_direction: torch.Tensor,
    baseline_target_area_index: int,
    baseline_aim_point: torch.Tensor,
    ablation_study_case: int,
    data_mappings: dict[str, list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
    | None = None,
    surface_reconstruction_optimization_configuration: dict[
        str, int | float | str | dict[str, float | int]
    ]
    | None = None,
    kinematic_reconstruction_optimization_configuration: dict[
        str, int | float | str | dict[str, float | int]
    ]
    | None = None,
    aimpoint_optimization_configuration: dict[
        str, int | float | str | dict[str, float | int]
    ]
    | None = None,
    target_distribution: torch.Tensor | None = None,
    data_for_stral_dir: pathlib.Path | None = None,
    device: torch.device | None = None,
) -> None:
    """
    Optimize the heliostat field with a combination of surface reconstruction, kinematic reconstruction and aim point optimization as part of an ablation study.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    results_dir : pathlib.Path
        Path to where the results are saved.
    results_number : int
        The current, incremented results number.
    baseline_incident_ray_direction : torch.Tensor
        Incident ray direction of the baseline measurement.
    baseline_target_area_index : int
        Target area index of the baseline measurement.
    baseline_aim_point : torch.Tensor
        Aim point of the baseline measurement.
    ablation_study_case : int
        Case number of the ablation study.
    data_mappings : dict[str, list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] | None
        Data mappings including heliostat names and their calibration files, for surface reconstruction, kinematic reconstruction
        and the evaluation plots.
    surface_reconstruction_optimization_configuration : dict[str, int | float | str | dict[str, float | int]] | None
        Configuration parameters for the surface reconstruction, if None no surfaces are reconstructed (default is None).
    kinematic_reconstruction_optimization_configuration : dict[str, int | float | str | dict[str, float | int]] | None
        Configuration parameters for the kinematic reconstruction, if None no kinematic is reconstructed (default is None).
    aimpoint_optimization_configuration : dict[str, int | float | str | dict[str, float | int]] | None
        Configuration parameters for the aim point optimization, if None no aim points are optimized (default is None).
    target_distribution : torch.Tensor | None
        Target distribution to aim for during aim point optimization (default is None).
    data_for_stral_dir : pathlib.Path | None
        Path to the directory where the data for the ``STRAL`` comparison is saved.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    runtime_log.info(f"case: {ablation_study_case}")
    device = get_device(device=device)

    results_path = results_dir / f"results_{results_number}.pt"
    results_dict: dict[str, dict[str, torch.Tensor]] = {}

    loaded = torch.load(results_path, weights_only=False)
    results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)

    if data_mappings is not None:
        data_parser = PaintCalibrationDataParser(
            sample_limit=4, centroid_extraction_method=paint_mappings.UTIS_KEY
        )
        data_kinematic_reconstruction: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: data_parser,
            config_dictionary.heliostat_data_mapping: data_mappings[
                "kinematic_reconstruction"
            ],
        }
        data_kinematic_plot: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: data_parser,
            config_dictionary.heliostat_data_mapping: data_mappings["kinematic_plot"],
        }
        data_surface_plot: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: data_parser,
            config_dictionary.heliostat_data_mapping: data_mappings["surface_plot"],
        }
        batch_size = 30 #380
        data_surfaces = create_surface_reconstruction_batches(
            data_mappings["surface_reconstruction"], data_parser, batch_size
        )

    surface_reconstruction_final_loss_per_heliostat = None
    kinematic_reconstruction_final_loss_per_heliostat = None
    aimpoint_optimization_final_loss = None
    motor_positions = None
    reconstructed_surface_path = (
        results_dir / f"reconstructed_surfaces_{results_number}.pt"
    )
    reconstructed_kinematic_path = (
        results_dir / f"reconstructed_kinematics_ideal_surface_{results_number}.pt"
    )

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        # TODO
        number_of_surface_points_per_facet = torch.tensor([50, 50], device=device)
        number_of_control_points_per_facet = torch.tensor([7, 7], device=device)
        number_of_rays_surface_reconstruction = 150
        number_of_rays = 4
        nurbs_degree = torch.tensor([3, 3], device=device)

        with h5py.File(scenario_path) as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )

            for heliostat_group in scenario.heliostat_field.heliostat_groups:
                heliostat_group.nurbs_degrees = nurbs_degree

            if surface_reconstruction_optimization_configuration is not None:
                if reconstructed_surface_path.exists():
                    print(
                        "A surface reconstruction viable for this case has been previously made. Loading the results."
                    )
                    loaded_nurbs_control_points = torch.load(
                        reconstructed_surface_path, weights_only=False
                    )
                    for heliostat_group, nurbs_control_points in zip(
                        scenario.heliostat_field.heliostat_groups,
                        loaded_nurbs_control_points,
                    ):
                        heliostat_group.nurbs_control_points = nurbs_control_points
                        evaluation_points = (
                            utils.create_nurbs_evaluation_grid(
                                number_of_evaluation_points=number_of_surface_points_per_facet,
                                device=device,
                            )
                            .unsqueeze(index_mapping.heliostat_dimension)
                            .unsqueeze(index_mapping.facet_index_unbatched)
                            .expand(
                                heliostat_group.number_of_heliostats,
                                heliostat_group.number_of_facets_per_heliostat,
                                -1,
                                -1,
                            )
                        )
                        nurbs_surfaces = NURBSSurfaces(
                            degrees=heliostat_group.nurbs_degrees,
                            control_points=heliostat_group.nurbs_control_points,
                            device=device,
                        )
                        (
                            new_surface_points,
                            new_surface_normals,
                        ) = nurbs_surfaces.calculate_surface_points_and_normals(
                            evaluation_points=evaluation_points,
                            canting=heliostat_group.canting,
                            facet_translations=heliostat_group.facet_translations,
                            device=device,
                        )
                        heliostat_group.surface_points = new_surface_points.reshape(
                            heliostat_group.active_surface_points.shape[
                                index_mapping.heliostat_dimension
                            ],
                            -1,
                            4,
                        )
                        heliostat_group.surface_normals = new_surface_normals.reshape(
                            heliostat_group.active_surface_normals.shape[
                                index_mapping.heliostat_dimension
                            ],
                            -1,
                            4,
                        )

                else:
                    surface_data_before = surface_data(
                        scenario=scenario,
                        ddp_setup=ddp_setup,
                        heliostat_data=data_surface_plot,
                        number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                        device=device,
                    )
                    surface_reconstruction_final_loss_per_heliostat = []
                    scenario.set_number_of_rays(
                        number_of_rays=number_of_rays_surface_reconstruction
                    )
                    for i, data in enumerate(data_surfaces):
                        print(f"batch no: {i}")
                        surface_reconstructor = SurfaceReconstructor(
                            ddp_setup=ddp_setup,
                            scenario=scenario,
                            data=data,
                            optimization_configuration=surface_reconstruction_optimization_configuration,
                            number_of_surface_points=number_of_surface_points_per_facet,
                            bitmap_resolution=torch.tensor([256, 256], device=device),
                            device=device,
                        )
                        losses_surfaces = surface_reconstructor.reconstruct_surfaces(
                            loss_definition=KLDivergenceLoss(), device=device
                        )
                        surface_reconstruction_final_loss_per_heliostat.append(
                            losses_surfaces
                        )
                        if ddp_setup["is_distributed"]:
                            torch.distributed.barrier()
                    surface_data_after = surface_data(
                        scenario=scenario,
                        ddp_setup=ddp_setup,
                        heliostat_data=data_surface_plot,
                        number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                        device=device,
                    )
                    merged_data_surface = merge_data(
                        unoptimized_data=surface_data_before,
                        optimized_data=surface_data_after,
                    )
                    results_dict["surface_reconstruction"] = merged_data_surface
                    torch.save(
                        [
                            heliostat_group.nurbs_control_points.detach()
                            for heliostat_group in scenario.heliostat_field.heliostat_groups
                        ],
                        reconstructed_surface_path,
                    )
                reconstructed_kinematic_path = (
                    results_dir
                    / f"reconstructed_kinematic_reconstructed_surfaces_{results_number}.pt"
                )

            if kinematic_reconstruction_optimization_configuration is not None:
                if reconstructed_kinematic_path.exists():
                    print(
                        "A kinematic reconstruction viable for this case has been previously made. Loading the results."
                    )
                    loaded_kinematics = torch.load(
                        reconstructed_kinematic_path, weights_only=False
                    )
                    for heliostat_group, loaded_kinematic in zip(
                        scenario.heliostat_field.heliostat_groups, loaded_kinematics
                    ):
                        heliostat_group.kinematic.rotation_deviation_parameters = (
                            loaded_kinematic["rotation_deviation_parameters"]
                        )
                        heliostat_group.kinematic.actuators.optimizable_parameters = (
                            loaded_kinematic["optimizable_parameters"]
                        )

                else:
                    bitmaps_for_kinematic_plots_before = kinematic_data(
                        scenario=scenario,
                        ddp_setup=ddp_setup,
                        heliostat_data=data_kinematic_plot,
                    )
                    scenario.set_number_of_rays(number_of_rays=number_of_rays)
                    kinematic_reconstructor = KinematicReconstructor(
                        ddp_setup=ddp_setup,
                        scenario=scenario,
                        data=data_kinematic_reconstruction,
                        optimization_configuration=kinematic_reconstruction_optimization_configuration,
                        reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
                    )
                    kinematic_reconstruction_final_loss_per_heliostat = (
                        kinematic_reconstructor.reconstruct_kinematic(
                            loss_definition=FocalSpotLoss(scenario=scenario),
                            device=device,
                        )
                    )
                    if ddp_setup["is_distributed"]:
                        torch.distributed.barrier()
                    bitmaps_for_kinematic_plots_after = kinematic_data(
                        scenario=scenario,
                        ddp_setup=ddp_setup,
                        heliostat_data=data_kinematic_plot,
                    )
                    merged_data_kinematic = merge_data(
                        unoptimized_data=bitmaps_for_kinematic_plots_before,
                        optimized_data=bitmaps_for_kinematic_plots_after,
                    )
                    surface = (
                        "ideal_surface"
                        if surface_reconstruction_optimization_configuration is None
                        else "reconstructed_surface"
                    )
                    results_dict[f"kinematic_reconstruction_{surface}"] = (
                        merged_data_kinematic
                    )
                    torch.save(
                        [
                            {
                                "rotation_deviation_parameters": heliostat_group.kinematic.rotation_deviation_parameters.detach(),
                                "optimizable_parameters": heliostat_group.kinematic.actuators.optimizable_parameters.detach(),
                            }
                            for heliostat_group in scenario.heliostat_field.heliostat_groups
                        ],
                        reconstructed_kinematic_path,
                    )

            if aimpoint_optimization_configuration is not None:
                scenario.set_number_of_rays(number_of_rays=number_of_rays)
                motor_positions_optimizer = MotorPositionsOptimizer(
                    ddp_setup=ddp_setup,
                    scenario=scenario,
                    optimization_configuration=aimpoint_optimization_configuration,
                    incident_ray_direction=baseline_incident_ray_direction,
                    target_area_index=baseline_target_area_index,
                    ground_truth=target_distribution,
                    bitmap_resolution=torch.tensor([256, 256]),
                    device=device,
                )
                aimpoint_optimization_final_loss = motor_positions_optimizer.optimize(
                    loss_definition=KLDivergenceLoss(), device=device
                )
                motor_positions = [
                    heliostat_group.kinematic.motor_positions
                    for heliostat_group in scenario.heliostat_field.heliostat_groups
                ]
                if ddp_setup["is_distributed"]:
                    torch.distributed.barrier()

            scenario.set_number_of_rays(number_of_rays=number_of_rays)
            combined_bitmaps_per_target = align_and_trace_rays(
                scenario=scenario,
                ddp_setup=ddp_setup,
                incident_ray_direction=baseline_incident_ray_direction,
                target_area_index=baseline_target_area_index,
                aim_point=baseline_aim_point,
                motor_positions=motor_positions,
            )

        if ablation_study_case == 7 and data_for_stral_dir is not None:
            save_heliostat_model(
                scenario=scenario,
                save_dir=data_for_stral_dir,
                results_number=results_number,
            )

        results_dict[f"ablation_study_case_{ablation_study_case}"] = {
            "flux": combined_bitmaps_per_target[baseline_target_area_index],
            "surface_reconstruction_loss_per_heliostat": surface_reconstruction_final_loss_per_heliostat,
            "kinematic_reconstruction_loss_per_heliostat": kinematic_reconstruction_final_loss_per_heliostat,
            "aimpoint_optimization_loss_per_heliostat": aimpoint_optimization_final_loss,
        }

        torch.save(results_dict, results_path)

        print(
            f"Ablation study case {ablation_study_case} results saved to {results_path}"
        )


@track_runtime(runtime_log)
def main() -> None:
    """
    Generate field optimization results and save them.

    This script performs ... TODO

    Parameters
    ----------
    config : str
        Path to the configuration file.
    data_dir : str
        Path to the data directory.
    device : str
        Device to use for the computation.
    results_dir : str
        Path to the directory for the results.
    scenarios_dir : str
        Path to the directory for saving the generated scenarios.
    """
    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default=default_config_path,
    )

    # Parse the config argument first to load the configuration.
    args, unknown = parser.parse_known_args()
    config_path = pathlib.Path(args.config)
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            warnings.warn(f"Error parsing YAML file: {exc}")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    device_default = config.get("device", "cuda")
    results_dir_default = config.get(
        "results_dir", "./examples/field_optimizations/results"
    )
    scenarios_dir_default = config.get(
        "scenarios_dir", "./examples/field_optimizations/scenarios"
    )
    measured_data_dir_default = config.get(
        "measured_data_dir", "./examples/field_optimizations/measured_data"
    )
    data_for_stral_dir_default = config.get(
        "data_for_stral_dir", "./examples/field_optimizations/data_for_stral"
    )
    heliostats_for_plots_default = config.get(
        "heliostats_for_plots", ["AK54", "AM55", "AM56"]
    )

    surface_reconstruction_optimization_configuration = config.get(
        "surface_reconstruction_optimization_configuration", {}
    )
    kinematic_reconstruction_optimization_configuration = config.get(
        "kinematic_reconstruction_optimization_configuration", {}
    )
    aimpoint_optimization_configuration = config.get(
        "aimpoint_optimization_configuration", {}
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results directory containing the viable heliostats list.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to the directory for saving the generated scenarios.",
        default=scenarios_dir_default,
    )
    parser.add_argument(
        "--measured_data_dir",
        type=str,
        help="Path to the directory containing measured data.",
        default=measured_data_dir_default,
    )
    parser.add_argument(
        "--data_for_stral_dir",
        type=str,
        help="Path to the directory where the data for the STRAL comparison will be saved.",
        default=data_for_stral_dir_default,
    )
    parser.add_argument(
        "--heliostats_for_plots",
        type=list[str],
        help="List of heliostat names used for the evaluation plots.",
        default=heliostats_for_plots_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    # for case in ["baseline", "full_field"]:
    for case in ["baseline"]:
        results_dir = pathlib.Path(args.results_dir) / f"{case}"
        results_dir.mkdir(parents=True, exist_ok=True)

        measured_data_dir = pathlib.Path(args.measured_data_dir)
        measured_data_dir.mkdir(parents=True, exist_ok=True)

        data_for_stral_dir = pathlib.Path(args.data_for_stral_dir) / case
        data_for_stral_dir.mkdir(parents=True, exist_ok=True)

        # Define scenario paths.
        scenario_path_ideal = (
            pathlib.Path(args.scenarios_dir) / f"ideal_{case}_scenario.h5"
        )
        if not scenario_path_ideal.exists():
            raise FileNotFoundError(
                f"The ideal scenario located at {scenario_path_ideal} could not be found! Please run the ``generate_scenarios.py`` to generate this scenario, or adjust the file path and try again."
            )
        scenario_path_deflectometry = (
            pathlib.Path(args.scenarios_dir)
            / "deflectometry_scenario_for_comparison.h5"
        )
        if not scenario_path_deflectometry.exists():
            raise FileNotFoundError(
                f"The deflectometry scenario located at {scenario_path_deflectometry} could not be found! Please run the ``generate_scenarios.py`` to generate this scenario, or adjust the file path and try again."
            )

        # Load viable heliostats data.
        viable_heliostats_data = (
            pathlib.Path(args.results_dir) / case / "viable_heliostats.json"
        )
        if not viable_heliostats_data.exists():
            raise FileNotFoundError(
                f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``generate_viable_heliostats_list.py`` script to generate this list, or adjust the file path and try again."
            )

        with open(viable_heliostats_data, "r") as f:
            viable_heliostats = json.load(f)

        runtime_log.info(f"Number of heliostats: {len(viable_heliostats)}")

        heliostat_data_mapping_kinematic_reconstruction: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ] = [
            (
                item["name"],
                [pathlib.Path(p) for p in item["calibrations"]],
                [pathlib.Path(p) for p in item["kinematic_reconstruction_flux_images"]],
            )
            for item in viable_heliostats
        ]

        heliostat_data_mapping_kinematic_plot: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ] = [
            (
                item["name"],
                [pathlib.Path(item["calibrations"][0])],
                [pathlib.Path(item["kinematic_reconstruction_flux_images"][0])],
            )
            for item in viable_heliostats
            if item["name"] in args.heliostats_for_plots
        ]

        heliostat_data_mapping_surface_reconstruction: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ] = [
            (
                item["name"],
                [pathlib.Path(p) for p in item["calibrations"]],
                [pathlib.Path(p) for p in item["surface_reconstruction_flux_images"]],
            )
            for item in viable_heliostats
        ]

        heliostat_data_mapping_surface_plot: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ] = [
            (
                item["name"],
                [pathlib.Path(item["calibrations"][0])],
                [pathlib.Path(item["surface_reconstruction_flux_images"][0])],
            )
            for item in viable_heliostats
            if item["name"] in args.heliostats_for_plots
        ]

        data_mappings = {
            "kinematic_reconstruction": heliostat_data_mapping_kinematic_reconstruction,
            "kinematic_plot": heliostat_data_mapping_kinematic_plot,
            "surface_reconstruction": heliostat_data_mapping_surface_reconstruction,
            "surface_plot": heliostat_data_mapping_surface_plot,
        }

        # TODO
        # Configure the optimizers and learning rate schedulers for all three optimization tasks.
        # ideal_surface_regularizer = IdealSurfaceRegularizer(
        #     weight=0.4,
        #     reduction_dimensions=(
        #         index_mapping.facet_dimension,
        #         index_mapping.points_dimension,
        #         index_mapping.coordinates_dimension,
        #     ),
        # )
        surface_reconstruction_optimization_configuration[
            config_dictionary.regularizers
        ] = [] #= [ideal_surface_regularizer]

        runtime_log.info(
            f"surface reconstruction: {surface_reconstruction_optimization_configuration}"
        )
        runtime_log.info(
            f"kinematic reconstruction: {kinematic_reconstruction_optimization_configuration}"
        )
        runtime_log.info(
            f"aim point optimization: {aimpoint_optimization_configuration}"
        )
        results_number = get_incremented_path_number(
            base_path=results_dir / "results.pt"
        )

        create_distributions(
            measured_data_dir=measured_data_dir,
            results_dir=results_dir,
            results_number=results_number,
            device=device,
        )

        create_deflectometry_surface_for_comparison(
            scenario_path=scenario_path_deflectometry,
            results_dir=results_dir,
            results_number=results_number,
            measured_data_dir=measured_data_dir,
            heliostats_for_plots=args.heliostats_for_plots,
            reconstruction_parameters={"number_of_surface_points": 50},
            device=device,
        )

        results_path = results_dir / f"results_{results_number}.pt"
        loaded = torch.load(results_path, weights_only=False)
        results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)

        baseline_incident_ray_direction = torch.nn.functional.normalize(
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
            - torch.tensor([-0.411, -0.706, +0.576, 1.0], device=device),
            dim=0,
        )
        baseline_target_area_index = 1
        baseline_aim_point = torch.tensor(
            [-0.19720931, -0.03458419, 5.4929e01, 1.0], device=device
        )
        target_distribution = results_dict["homogeneous_distribution"]

        # Ablation study cases:
        # 1. Surface: ideal             Kinematic: ideal            Aim Points: center
        # 2. Surface: ideal             Kinematic: ideal            Aim Points: optimized
        # 3. Surface: ideal             Kinematic: reconstructed    Aim Points: center
        # 4. Surface: ideal             Kinematic: reconstructed    Aim Points: optimized
        # 5. Surface: reconstructed     Kinematic: ideal            Aim Points: center
        # 6. Surface: reconstructed     Kinematic: ideal            Aim Points: optimized
        # 7. Surface: reconstructed     Kinematic: reconstructed    Aim Points: center
        # 8. Surface: reconstructed     Kinematic: reconstructed    Aim Points: optimized

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=1,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=2,
            aimpoint_optimization_configuration=aimpoint_optimization_configuration,
            target_distribution=target_distribution,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=3,
            data_mappings=data_mappings,
            kinematic_reconstruction_optimization_configuration=kinematic_reconstruction_optimization_configuration,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=4,
            data_mappings=data_mappings,
            kinematic_reconstruction_optimization_configuration=kinematic_reconstruction_optimization_configuration,
            aimpoint_optimization_configuration=aimpoint_optimization_configuration,
            target_distribution=target_distribution,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=5,
            data_mappings=data_mappings,
            surface_reconstruction_optimization_configuration=surface_reconstruction_optimization_configuration,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=6,
            data_mappings=data_mappings,
            surface_reconstruction_optimization_configuration=surface_reconstruction_optimization_configuration,
            aimpoint_optimization_configuration=aimpoint_optimization_configuration,
            target_distribution=target_distribution,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=7,
            data_mappings=data_mappings,
            surface_reconstruction_optimization_configuration=surface_reconstruction_optimization_configuration,
            kinematic_reconstruction_optimization_configuration=kinematic_reconstruction_optimization_configuration,
            data_for_stral_dir=data_for_stral_dir,
            device=device,
        )

        ablation_study(
            scenario_path=scenario_path_ideal,
            results_dir=results_dir,
            results_number=results_number,
            baseline_incident_ray_direction=baseline_incident_ray_direction,
            baseline_target_area_index=baseline_target_area_index,
            baseline_aim_point=baseline_aim_point,
            ablation_study_case=8,
            data_mappings=data_mappings,
            surface_reconstruction_optimization_configuration=surface_reconstruction_optimization_configuration,
            kinematic_reconstruction_optimization_configuration=kinematic_reconstruction_optimization_configuration,
            aimpoint_optimization_configuration=aimpoint_optimization_configuration,
            target_distribution=target_distribution,
            device=device,
        )


if __name__ == "__main__":
    runtime_log.info("-----------------")
    main()
    runtime_log.info("\n\n")
