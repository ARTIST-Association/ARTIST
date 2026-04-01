import argparse
import csv
import gc
import json
import pathlib
import re
import sys
import warnings
from typing import Any, cast

import h5py
from matplotlib import pyplot as plt
import paint.util.paint_mappings as paint_mappings
import torch
import yaml


from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematics_reconstructor import KinematicsReconstructor
from artist.core.loss_functions import FocalSpotLoss, KLDivergenceLoss
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
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


def create_distributions(
    measured_data_dir: pathlib.Path,
    results_path: pathlib.Path,
    device: torch.device | None = None,
) -> None:
    """
    Save the baseline measured distribution and a homogeneous distribution in the results file.

    Parameters
    ----------
    measured_data_dir : pathlib.Path
        Path to the measured baseline data.
    results_path : pathlib.Path
        Path to where the results are saved.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device=device)

    results_dict: dict[str, dict[str, torch.Tensor]] = {}

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
            total_width=256, slope_width=30, plateau_width=180, device=device #180
        )
        u_trapezoid = utils.trapezoid_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        eu_trapezoid = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

        homogeneous_distribution = eu_trapezoid / eu_trapezoid.sum()

        results_dict["homogeneous_distribution"] = homogeneous_distribution

    torch.save(results_dict, results_path)


def create_deflectometry_surface_for_comparison(
    scenario_path: pathlib.Path,
    results_path: pathlib.Path,
    measured_data_dir: pathlib.Path,
    heliostats_for_plots: list[str],
    number_of_surface_points: dict[str, float | int],
    device: torch.device | None,
) -> None:
    """
    Create the surface from the measured deflectometry as comparison.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    results_path : pathlib.Path
        Path to where the results are saved.
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
                number_of_surface_points,
                number_of_surface_points,
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
    with torch.no_grad():
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
            data["mirror_offsets"].extend(
                heliostat_group.kinematics.translation_deviation_parameters[:, 7].tolist()
            )
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
    with torch.no_grad():
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


def kinematics_plots(
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
    Extract heliostat kinematics information.

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
        Dictionary containing kinematics data per heliostat.
    """
    with torch.no_grad():
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
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                device=device,
            )
            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask.detach(), device=device
            )
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.solar_tower.get_centers_of_target_areas(target_area_indices=target_area_indices, device=device).detach(),
                incident_ray_directions=incident_ray_directions.detach(),
                active_heliostats_mask=active_heliostats_mask.detach(),
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
            bitmaps_per_heliostat, _, _, _ = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions.detach(),
                active_heliostats_mask=active_heliostats_mask.detach(),
                target_area_indices=target_area_indices.detach(),
                device=device,
            )
            names = [
                heliostat_group.names[i]
                for i in torch.nonzero(active_heliostats_mask).squeeze()
            ]

            for i, heliostat in enumerate(names):
                bitmaps_for_plots[heliostat] = {
                    "artist_flux": bitmaps_per_heliostat[i].detach(),
                    "measured_flux": measured_fluxes[i].detach(),
                }

        return bitmaps_for_plots

def surface_plots(
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
    with torch.no_grad():
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
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                device=device,
            )
            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask.detach(), device=device
            )
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.solar_tower.get_centers_of_target_areas(target_area_indices=target_area_indices, device=device).detach(),
                incident_ray_directions=incident_ray_directions.detach(),
                active_heliostats_mask=active_heliostats_mask.detach(),
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
            bitmaps_per_heliostat, _, _, _= ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions.detach(),
                active_heliostats_mask=active_heliostats_mask.detach(),
                target_area_indices=target_area_indices.detach(),
                device=device,
            )
            cropped_flux_distributions = utils.crop_flux_distributions_around_center(
                flux_distributions=bitmaps_per_heliostat.detach(),
                crop_width=config_dictionary.utis_crop_width,
                crop_height=config_dictionary.utis_crop_height,
                target_plane_widths=scenario.target_areas.dimensions[target_area_indices][
                    :, index_mapping.target_area_width
                ].detach(),
                target_plane_heights=scenario.target_areas.dimensions[target_area_indices][
                    :, index_mapping.target_area_height
                ].detach(),
                device=device,
            )
            names = [
                heliostat_group.names[i]
                for i in torch.nonzero(active_heliostats_mask.detach()).squeeze()
            ]

            for index, heliostat in enumerate(names):
                heliostat_index_global = heliostat_group.names.index(heliostat)
                data_for_plots[heliostat] = {
                    "measured_flux": measured_fluxes[index].detach(),
                    "artist_flux": cropped_flux_distributions[index].detach(),
                    "surface_points": heliostat_group.surface_points[heliostat_index_global].detach(),
                    "surface_normals": heliostat_group.surface_normals[heliostat_index_global].detach(),
                    "canting": heliostat_group.canting[heliostat_index_global].detach(),
                    "facet_translations": heliostat_group.facet_translations[heliostat_index_global].detach(),
                }

        return data_for_plots


def aim_point_plots(
    scenario: Scenario,
    incident_ray_direction: torch.Tensor,
    target_area_index: int,
    aim_point,
    dni: float,
    id: str,
    batch_size: int = 96,
    number_of_rays = 25,
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Extract heliostat kinematics information.

    Parameters
    ----------
    scenario : Scenario
        The scenario.
    incident_ray_direction : torch.Tensor
        The incident ray direction during the optimization.
        Tensor of shape [4].
    target_area_index : int
        The index of the target used for the optimization.
    dni : float
        Direct normal irradiance in W/m^2.
    id : str
        Identifier fluxes.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        Kinematics data per heliostat.
    """
    with torch.no_grad():
        device = get_device(device)
        bitmap_resolution = torch.tensor([256, 256], device=device)
        total_flux = torch.zeros(
            (
                bitmap_resolution[index_mapping.unbatched_bitmap_e],
                bitmap_resolution[index_mapping.unbatched_bitmap_u],
            ),
            device=device,
        )

        for heliostat_group_index, heliostat_group in enumerate(
            scenario.heliostat_field.heliostat_groups
        ):
            (active_heliostats_mask, target_area_indices, incident_ray_directions) = (
                scenario.index_mapping(
                    heliostat_group=heliostat_group,
                    single_incident_ray_direction=incident_ray_direction,
                    single_target_area_index=target_area_index,
                    device=device,
                )
            )
            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )
            if id == "before":
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=aim_point,
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )
            elif id == "after":
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=heliostat_group.kinematics.active_motor_positions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

        for heliostat_group_index, heliostat_group in enumerate(
            scenario.heliostat_field.heliostat_groups
        ):
            (active_heliostats_mask, target_area_indices, incident_ray_directions) = (
                scenario.index_mapping(
                    heliostat_group=heliostat_group,
                    single_incident_ray_direction=incident_ray_direction,
                    single_target_area_index=target_area_index,
                    device=device,
                )
            )
            scenario.set_number_of_rays(number_of_rays=number_of_rays)
            ray_tracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                blocking_active=True,
                batch_size=batch_size,
                bitmap_resolution=bitmap_resolution,
                dni=dni,
            )
            bitmaps_per_heliostat, _, _, _ = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_indices=target_area_indices,
                device=device,
            )
            # Uncomment for single heliostat flux analysis.
            # for i in range(bitmaps_per_heliostat.shape[0]):
            #     plt.imshow(bitmaps_per_heliostat[i].cpu().detach(), cmap="gray")
            #     plt.savefig(f"bitmaps/aim_points/{heliostat_group.names[i]}.png")
            #     plt.close()
            flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                bitmaps_per_heliostat=bitmaps_per_heliostat,
                target_area_indices=target_area_indices,
                device=device,
            )[target_area_index]
            total_flux += flux_distribution_on_target

        return total_flux

@track_runtime(runtime_log)
def full_field_optimizations(
    scenario_path: pathlib.Path,
    results_path: pathlib.Path,
    basic_config: dict[str, Any],
    data_mappings: dict[str, Any] | None = None,
    surface_config: dict[str, Any] | None = None,
    kinematics_config: dict[str, Any] | None = None,
    aim_point_config: dict[str, Any] | None = None,
    target_distribution: torch.Tensor | None = None,
    data_for_stral_dir: pathlib.Path | None = None,
    device: torch.device | None = None,
) -> None:
    """
    Optimize the heliostat field with a combination of surface reconstruction, kinematics reconstruction and aim point optimization as part of an ablation study.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    results_path : pathlib.Path
        Path to where the results are saved.
    ablation_study_case : int
        Case number of the ablation study.
    basic_config : dict[str, Any]
        Configuration for number of surface points, number of rays and baseline data. 
    data_mappings : dict[str, list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] | None
        Data mappings including heliostat names and their calibration files, for surface reconstruction, kinematics reconstruction
        and the evaluation plots.
    surface_config : dict[str, int | float | str | dict[str, float | int]] | None
        Configuration parameters for the surface reconstruction, if None no surfaces are reconstructed (default is None).
    kinematics_config : dict[str, int | float | str | dict[str, float | int]] | None
        Configuration parameters for the kinematics reconstruction, if None no kinematics is reconstructed (default is None).
    aim_point_config : dict[str, int | float | str | dict[str, float | int]] | None
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
    device = get_device(device=device)

    results_number = (
        int(m.group(1))
        if (m := re.match(r"results_(\d+)\.pt", results_path.name))
        else 0
    )
    results_dict = cast(
        dict[str, dict[str, torch.Tensor]], torch.load(results_path, weights_only=False)
    )

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        bitmap_resolution = torch.tensor([256, 256], device=device)
        baseline_incident_ray_direction = torch.nn.functional.normalize(torch.tensor(basic_config["baseline_incident_ray_direction"], device=device), dim=0)
        baseline_target_area = basic_config["baseline_target_area"]
        baseline_aim_point = torch.tensor(basic_config["baseline_aim_point"], device=device)
        baseline_dni=basic_config["dni"]

        number_of_surface_points_per_facet = torch.tensor(
            [surface_config["number_of_surface_points"], surface_config["number_of_surface_points"]], 
            device=device
        )
        number_of_control_points_per_facet = torch.tensor(
            [surface_config["number_of_control_points"], surface_config["number_of_control_points"]], 
            device=device
        )
        nurbs_degree = torch.tensor(
            [surface_config["nurbs_degree"], surface_config["nurbs_degree"]], 
            device=device
        )

        with h5py.File(scenario_path) as scenario_file:
            scenario_kinematics_ideal_surfaces = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )
        for heliostat_group in scenario_kinematics_ideal_surfaces.heliostat_field.heliostat_groups:
            heliostat_group.nurbs_degrees = nurbs_degree

        aim_point_data_ideal_models = aim_point_plots(
            scenario=scenario_kinematics_ideal_surfaces,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_kinematics_ideal_surfaces.solar_tower.target_name_to_index[baseline_target_area],
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="before",
            batch_size=20,
            number_of_rays=200,
            device=device
        )
        results_dict["ideal_model"] = {
            "aim_point_plot": aim_point_data_ideal_models
        } 
        torch.cuda.empty_cache()
        
        kinematics_data_before = kinematics_plots(
            scenario=scenario_kinematics_ideal_surfaces,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["kinematics_plot"],
            device=device
        )
        scenario_kinematics_ideal_surfaces.set_number_of_rays(number_of_rays=kinematics_config["number_of_rays"])
        optimization_configuration_kinematics = {
            config_dictionary.optimization: {
                config_dictionary.initial_learning_rate: kinematics_config["initial_learning_rate"],
                config_dictionary.tolerance: 1e-5,
                config_dictionary.max_epoch: kinematics_config["max_epoch"],
                config_dictionary.batch_size: kinematics_config["batch_size"],
                config_dictionary.log_step: 10,
                config_dictionary.early_stopping_delta: 1e-12,
                config_dictionary.early_stopping_patience: 10000,
                config_dictionary.early_stopping_window: 10000,
            },
            config_dictionary.scheduler: {
                config_dictionary.scheduler_type: kinematics_config["scheduler"],
                config_dictionary.gamma: kinematics_config["gamma"],
                config_dictionary.min: kinematics_config["min_learning_rate"],
                config_dictionary.max: kinematics_config["max_learning_rate"],
                config_dictionary.step_size_up:kinematics_config["step_size_up"],
                config_dictionary.reduce_factor: kinematics_config["reduce_factor"],
                config_dictionary.patience: kinematics_config["patience"],
                config_dictionary.threshold: kinematics_config["threshold"],
                config_dictionary.cooldown: kinematics_config["cooldown"],
            }
        }
        kinematics_reconstructor = KinematicsReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario_kinematics_ideal_surfaces,
            data=data_mappings["kinematics_reconstruction"],
            dni=baseline_dni,
            optimization_configuration=optimization_configuration_kinematics,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        )
        kinematics_reconstruction_final_loss_per_heliostat, loss_history_kinematics = (
            kinematics_reconstructor.reconstruct_kinematics(
                loss_definition=FocalSpotLoss(scenario=scenario_kinematics_ideal_surfaces),
                device=device,
            )
        )
        if ddp_setup["is_distributed"]:
            torch.distributed.barrier()
        kinematics_data_after = kinematics_plots(
            scenario=scenario_kinematics_ideal_surfaces,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["kinematics_plot"],
            device=device
        )
        merged_data_kinematics = merge_data(
            unoptimized_data=kinematics_data_before,
            optimized_data=kinematics_data_after,
        )
        aim_point_data_kinematic_reconstruction = aim_point_plots(
            scenario=scenario_kinematics_ideal_surfaces,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_kinematics_ideal_surfaces.target_areas.names.index(baseline_target_area),
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="before",
            batch_size=20,
            number_of_rays=200,
            device=device
        )
        results_dict["kinematics_reconstruction_with_ideal_surfaces"] = {
            "flux_plot_data": merged_data_kinematics,
            "loss_history": loss_history_kinematics,
            "loss": kinematics_reconstruction_final_loss_per_heliostat,
            "aim_point_plot": aim_point_data_kinematic_reconstruction
        }
        torch.cuda.empty_cache()

        # Surface reconstruction.
        with h5py.File(scenario_path) as scenario_file:
            scenario_surface = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )
        for heliostat_group in scenario_surface.heliostat_field.heliostat_groups:
            heliostat_group.nurbs_degrees = nurbs_degree
        surface_data_before = surface_plots(
            scenario=scenario_surface,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["surface_plot"],
            device=device,
        )
        scenario_surface.set_number_of_rays(
            number_of_rays=surface_config["number_of_rays"]
        )
        surface_reconstruction_final_loss_per_heliostat = []
        loss_history_surface = []
        data_surfaces = []
        batch_size = surface_config["batch_size_outer"]
        heliostat_data = data_mappings["surface_reconstruction"][
            config_dictionary.heliostat_data_mapping
        ]
        for i in range(0, len(heliostat_data), batch_size):
            batch = heliostat_data[i : i + batch_size]
            data_surfaces.append(
                {
                    config_dictionary.data_parser: data_mappings[
                        "surface_reconstruction"
                    ][config_dictionary.data_parser],
                    config_dictionary.heliostat_data_mapping: batch,
                }
            )
        optimization_configuration_surface = {
            config_dictionary.optimization: {
                config_dictionary.initial_learning_rate: surface_config["initial_learning_rate"],
                config_dictionary.tolerance: 1e-5,
                config_dictionary.max_epoch: surface_config["max_epoch"],
                config_dictionary.batch_size: surface_config["batch_size"],
                config_dictionary.log_step: 10,
                config_dictionary.early_stopping_delta: 1e-12,
                config_dictionary.early_stopping_patience: 10000,
                config_dictionary.early_stopping_window: 10000,
            },
            config_dictionary.scheduler: {
                config_dictionary.scheduler_type: surface_config["scheduler"],
                config_dictionary.gamma: surface_config["gamma"],
                config_dictionary.min: surface_config["min_learning_rate"],
                config_dictionary.max: surface_config["max_learning_rate"],
                config_dictionary.step_size_up:surface_config["step_size_up"],
                config_dictionary.reduce_factor: surface_config["reduce_factor"],
                config_dictionary.patience: surface_config["patience"],
                config_dictionary.threshold: surface_config["threshold"],
                config_dictionary.cooldown: surface_config["cooldown"],
            },
            config_dictionary.constraints:{
                config_dictionary.weight_smoothness: surface_config["weight_smoothness"],
                config_dictionary.weight_ideal_surface: surface_config["weight_ideal_surface"],
                config_dictionary.rho_flux_integral: surface_config["rho_flux_integral"],
                config_dictionary.energy_tolerance: surface_config["energy_tolerance"],
            }
        }
        for data in data_surfaces:
            surface_reconstructor = SurfaceReconstructor(
                ddp_setup=ddp_setup,
                scenario=scenario_surface,
                data=data,
                optimization_configuration=optimization_configuration_surface,
                dni=baseline_dni,
                number_of_surface_points=number_of_surface_points_per_facet,
                bitmap_resolution=bitmap_resolution,
                device=device,
            )
            losses_surfaces, loss_history_surface_part = surface_reconstructor.reconstruct_surfaces(
                loss_definition=KLDivergenceLoss(), device=device
            )
            surface_reconstruction_final_loss_per_heliostat.append(
                losses_surfaces
            )
            loss_history_surface.append(loss_history_surface_part)
            if ddp_setup["is_distributed"]:
                torch.distributed.barrier()
        surface_data_after = surface_plots(
            scenario=scenario_surface,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["surface_plot"],
            device=device,
        )
        merged_data_surface = merge_data(
            unoptimized_data=surface_data_before,
            optimized_data=surface_data_after,
        )
        reconstructed_nurbs_control_points = [
            heliostat_group.nurbs_control_points.detach()
            for heliostat_group in scenario_surface.heliostat_field.heliostat_groups
        ]
        aim_point_data_surface_reconstruction = aim_point_plots(
            scenario=scenario_surface,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_surface.target_areas.names.index(baseline_target_area),
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="before",
            batch_size=20,
            number_of_rays=200,
            device=device
        )
        results_dict["surface_reconstruction"] = {
            "flux_plot_data": merged_data_surface,
            "loss_history": loss_history_surface,
            "loss": surface_reconstruction_final_loss_per_heliostat,
            "aim_point_plot": aim_point_data_surface_reconstruction
        } 
        torch.cuda.empty_cache()

        # Kinematics reconstruction with reconstructed surfaces.
        with h5py.File(scenario_path) as scenario_file:
            scenario_kinematics = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )
        for heliostat_group, control_points in zip(
            scenario_kinematics.heliostat_field.heliostat_groups,
            reconstructed_nurbs_control_points,
        ):
            heliostat_group.nurbs_degrees=nurbs_degree
            heliostat_group.nurbs_control_points = control_points
        scenario_kinematics.heliostat_field.update_surfaces(device=device)
        kinematics_data_before = kinematics_plots(
            scenario=scenario_kinematics,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["kinematics_plot"],
            device=device
        )
        scenario_kinematics.set_number_of_rays(number_of_rays=kinematics_config["number_of_rays"])
        optimization_configuration_kinematics = {
            config_dictionary.optimization: {
                config_dictionary.initial_learning_rate: kinematics_config["initial_learning_rate"],
                config_dictionary.tolerance: 1e-5,
                config_dictionary.max_epoch: kinematics_config["max_epoch"],
                config_dictionary.batch_size: kinematics_config["batch_size"],
                config_dictionary.log_step: 10,
                config_dictionary.early_stopping_delta: 1e-12,
                config_dictionary.early_stopping_patience: 10000,
                config_dictionary.early_stopping_window: 10000,
            },
            config_dictionary.scheduler: {
                config_dictionary.scheduler_type: kinematics_config["scheduler"],
                config_dictionary.gamma: kinematics_config["gamma"],
                config_dictionary.min: kinematics_config["min_learning_rate"],
                config_dictionary.max: kinematics_config["max_learning_rate"],
                config_dictionary.step_size_up:kinematics_config["step_size_up"],
                config_dictionary.reduce_factor: kinematics_config["reduce_factor"],
                config_dictionary.patience: kinematics_config["patience"],
                config_dictionary.threshold: kinematics_config["threshold"],
                config_dictionary.cooldown: kinematics_config["cooldown"],
            }
        }
        kinematics_reconstructor = KinematicsReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario_kinematics,
            data=data_mappings["kinematics_reconstruction"],
            dni=baseline_dni,
            optimization_configuration=optimization_configuration_kinematics,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        )
        kinematics_reconstruction_final_loss_per_heliostat, loss_history_kinematics = (
            kinematics_reconstructor.reconstruct_kinematics(
                loss_definition=FocalSpotLoss(scenario=scenario_kinematics),
                device=device,
            )
        )
        if ddp_setup["is_distributed"]:
            torch.distributed.barrier()
        kinematics_data_after = kinematics_plots(
            scenario=scenario_kinematics,
            ddp_setup=ddp_setup,
            heliostat_data=data_mappings["kinematics_plot"],
            device=device
        )
        merged_data_kinematics = merge_data(
            unoptimized_data=kinematics_data_before,
            optimized_data=kinematics_data_after,
        )
        reconstructed_kinematics = [
            {
                "rotation_deviation_parameters": heliostat_group.kinematics.rotation_deviation_parameters.detach(),
                "optimizable_parameters": heliostat_group.kinematics.actuators.optimizable_parameters.detach(),
            }
            for heliostat_group in scenario_kinematics.heliostat_field.heliostat_groups
        ]
        aim_point_data_combined_reconstruction = aim_point_plots(
            scenario=scenario_kinematics,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_kinematics.target_areas.names.index(baseline_target_area),
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="before",
            batch_size=20,
            number_of_rays=200,
            device=device
        )
        results_dict["kinematics_reconstruction_with_reconstructed_surfaces"] = {
            "flux_plot_data": merged_data_kinematics,
            "loss_history": loss_history_kinematics,
            "loss": kinematics_reconstruction_final_loss_per_heliostat,
            "aim_point_plot": aim_point_data_combined_reconstruction
        }
        torch.cuda.empty_cache()

        # Aim point optimization.
        with h5py.File(scenario_path) as scenario_file:
            scenario_aim_points = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )
        for heliostat_group, control_points, kinematics in zip(
            scenario_aim_points.heliostat_field.heliostat_groups,
            reconstructed_nurbs_control_points,
            reconstructed_kinematics
            
        ):
            heliostat_group.nurbs_control_points = control_points
            heliostat_group.nurbs_degrees = nurbs_degree
            heliostat_group.kinematics.rotation_deviation_parameters = kinematics[
                "rotation_deviation_parameters"
            ]
            heliostat_group.kinematics.actuators.optimizable_parameters = kinematics[
                "optimizable_parameters"
            ]
        scenario_aim_points.heliostat_field.update_surfaces(device=device)
        aim_points_data_before = aim_point_plots(
            scenario=scenario_aim_points,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_aim_points.target_areas.names.index(baseline_target_area),
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="before",
            batch_size=20,
            number_of_rays=200,
            device=device,
        )  
        scenario_aim_points.set_number_of_rays(number_of_rays=aim_point_config["number_of_rays"])
        optimization_configuration_aim_points = {
            config_dictionary.optimization: {
                config_dictionary.initial_learning_rate: aim_point_config["initial_learning_rate"],
                config_dictionary.tolerance: 1e-5,
                config_dictionary.max_epoch: aim_point_config["max_epoch"],
                config_dictionary.batch_size: aim_point_config["batch_size"],
                config_dictionary.log_step: 10,
                config_dictionary.early_stopping_delta: 1e-12,
                config_dictionary.early_stopping_patience: 10000,
                config_dictionary.early_stopping_window: 10000,
            },
            config_dictionary.scheduler: {
                config_dictionary.scheduler_type: aim_point_config["scheduler"],
                config_dictionary.gamma: aim_point_config["gamma"],
                config_dictionary.min: aim_point_config["min_learning_rate"],
                config_dictionary.max: aim_point_config["max_learning_rate"],
                config_dictionary.step_size_up:aim_point_config["step_size_up"],
                config_dictionary.reduce_factor: aim_point_config["reduce_factor"],
                config_dictionary.patience: aim_point_config["patience"],
                config_dictionary.threshold: aim_point_config["threshold"],
                config_dictionary.cooldown: aim_point_config["cooldown"],
            },
            config_dictionary.constraints:{
                config_dictionary.rho_flux_integral: aim_point_config["rho_flux_integral"],
                config_dictionary.rho_local_flux: aim_point_config["rho_local_flux"],
                config_dictionary.rho_spillage: aim_point_config["rho_spillage"],
                config_dictionary.max_flux_density: aim_point_config["max_flux_density"],   
            }
        }
        motor_positions_optimizer = MotorPositionsOptimizer(
            ddp_setup=ddp_setup,
            scenario=scenario_aim_points,
            optimization_configuration=optimization_configuration_aim_points,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_aim_points.target_areas.names.index(baseline_target_area),
            ground_truth=target_distribution,
            dni=baseline_dni,
            bitmap_resolution=bitmap_resolution,
            device=device,
        )
        aimpoint_optimization_final_loss, loss_history_aim_points, _, _ = motor_positions_optimizer.optimize(
            loss_definition=KLDivergenceLoss(), device=device
        )
        aim_point_data_after = aim_point_plots(
            scenario=scenario_aim_points,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario_aim_points.target_areas.names.index(baseline_target_area),
            aim_point=baseline_aim_point,
            dni=baseline_dni,
            id="after",
            batch_size=20,
            number_of_rays=200,
            device=device,
        )        
        if ddp_setup["is_distributed"]:
            torch.distributed.barrier()
        results_dict["aim_point_optimization_reconstructed_model"] = {
            "aim_point_plot": torch.stack((aim_points_data_before, aim_point_data_after)),
            "loss_history": loss_history_aim_points,
            "loss": aimpoint_optimization_final_loss,
        }

        # Save data to be used in the STRAL comparison.
        if data_for_stral_dir is not None:
            save_heliostat_model(
                scenario=scenario_aim_points,
                save_dir=data_for_stral_dir,
                results_number=results_number,
            )
        
        torch.save(results_dict, results_path)


def create_heliostat_data_mappings(
    viable_heliostats_data: pathlib.Path, 
    heliostats_for_plots: list[str],
    sample_limit_surfaces: int,
    sample_limit_kinematics: int,
) -> dict[str, Any]:
    """
    Create the data mappings for the heliostats in the different optimization tasks.

    Parameters
    ----------
    viable_heliostats_data : pathlib.Path
        Path to the viable heliostats list.
    heliostats_for_plots : list[str]
        List of all heliostats considered in the plots.

    Returns
    -------
    dict[str, Any]
        The mappings from heliostat name to data files and data parsers for each task.
    """
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)
    
    data_parser_surface = PaintCalibrationDataParser(
        sample_limit=sample_limit_surfaces, centroid_extraction_method=paint_mappings.UTIS_KEY
    )
    data_parser_kinematics = PaintCalibrationDataParser(
        sample_limit=sample_limit_kinematics, centroid_extraction_method=paint_mappings.UTIS_KEY
    )
    data_parser_plots = PaintCalibrationDataParser(
        sample_limit=2, centroid_extraction_method=paint_mappings.UTIS_KEY
    )

    # Data mappings for the kinematics reconstruction plot.
    path_mapping_kinematics_plot: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ] = [
        (
            item["name"],
            [pathlib.Path(item["calibrations"][0])],
            [pathlib.Path(item["kinematic_reconstruction_flux_images"][0])],
        )
        for item in viable_heliostats
        if item["name"] in heliostats_for_plots
    ]
    data_kinematics_plot: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser_plots,
        config_dictionary.heliostat_data_mapping: path_mapping_kinematics_plot,
    }

    # Data mappings for the kinematics reconstruction.
    path_mapping_kinematics_reconstruction: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["kinematic_reconstruction_flux_images"]],
        )
        for item in viable_heliostats
    ]
    data_kinematics_reconstruction: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser_kinematics,
        config_dictionary.heliostat_data_mapping: path_mapping_kinematics_reconstruction,
    }

    # Data mappings for the surface reconstruction plot.
    path_mapping_surface_plot: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ] = [
        (
            item["name"],
            [pathlib.Path(item["calibrations"][0])],
            [pathlib.Path(item["surface_reconstruction_flux_images"][0])],
        )
        for item in viable_heliostats
        if item["name"] in heliostats_for_plots
    ]
    data_surface_plot: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser_plots,
        config_dictionary.heliostat_data_mapping: path_mapping_surface_plot,
    }

    # Data mappings for the surface reconstruction.
    path_mapping_surface_reconstruction: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["surface_reconstruction_flux_images"]],
        )
        for item in viable_heliostats
    ]
    data_surface_reconstruction: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser_surface,
        config_dictionary.heliostat_data_mapping: path_mapping_surface_reconstruction,
    }

    data_mappings = {
        "kinematics_reconstruction": data_kinematics_reconstruction,
        "kinematics_plot": data_kinematics_plot,
        "surface_reconstruction": data_surface_reconstruction,
        "surface_plot": data_surface_plot,
    }

    return data_mappings


def parse_runtimes(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "main started" in lines[i]:
            start_idx = i
            break

    last_run = lines[start_idx:]
    ablation_times = []
    results_file_number = None
    number_of_heliostats = None

    for line in last_run:
        if "Results_file_number:" in line:
            match = re.search(r"Results_file_number:\s*(\d+)", line)
            if match:
                results_file_number = int(match.group(1))

        elif "Number of heliostats:" in line:
            match = re.search(r"Number of heliostats:\s*(\d+)", line)
            if match:
                number_of_heliostats = int(match.group(1))

        elif "ablation_study finished in" in line:
            match = re.search(r"finished in ([\d\.]+)s", line)
            if match:
                ablation_times.append(float(match.group(1)))
            
    return {
        "runtimes": ablation_times,
        "run_id": (results_file_number, number_of_heliostats)
    }


@track_runtime(runtime_log)
def main() -> None:
    """Generate field optimization results and save them."""
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
    surface_reconstruction_optimization_configuration_default = config.get(
        "surface_reconstruction_optimization_configuration", {}
    )
    kinematics_reconstruction_optimization_configuration_default = config.get(
        "kinematics_reconstruction_optimization_configuration", {}
    )
    aim_point_optimization_configuration_default = config.get(
        "aim_point_optimization_configuration", {}
    )
    basic_config_default = config.get(
        "basic_config", {}
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
    parser.add_argument(
        "--surface_reconstruction_optimization_configuration",
        type=dict[Any],
        help="Config.",
        default=surface_reconstruction_optimization_configuration_default,
    )
    parser.add_argument(
        "--kinematics_reconstruction_optimization_configuration",
        type=dict[Any],
        help="Config.",
        default=kinematics_reconstruction_optimization_configuration_default,
    )
    parser.add_argument(
        "--aim_point_optimization_configuration",
        type=dict[Any],
        help="Config.",
        default=aim_point_optimization_configuration_default,
    )
    parser.add_argument(
        "--basic_config",
        type=dict[Any],
        help="Config.",
        default=basic_config_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)
    device = get_device(torch.device(args.device))

    #for case in ["baseline", "full_field"]:
    for case in ["baseline"]:
        # Set directory paths.
        results_dir = pathlib.Path(args.results_dir) / f"{case}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_number = get_incremented_path_number(
            base_path=results_dir / "results.pt"
        )
        results_path = results_dir / f"results_{results_number}.pt"

        measured_data_dir = pathlib.Path(args.measured_data_dir)
        measured_data_dir.mkdir(parents=True, exist_ok=True)

        data_for_stral_dir = pathlib.Path(args.data_for_stral_dir) / case
        data_for_stral_dir.mkdir(parents=True, exist_ok=True)

        surface_optimization_config = args.surface_reconstruction_optimization_configuration
        kinematics_optimization_config = args.kinematics_reconstruction_optimization_configuration
        aim_point_optimization_config = args.aim_point_optimization_configuration
        basic_config = args.basic_config

        # Define scenario paths and viable heliostat data paths.
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

        viable_heliostats_data = (
            pathlib.Path(args.results_dir) / case / "viable_heliostats.json"
        )
        if not viable_heliostats_data.exists():
            raise FileNotFoundError(
                f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``generate_viable_heliostats_list.py`` script to generate this list, or adjust the file path and try again."
            )

        # Create heliostat data mappings from viable heliostats.
        data_mappings = create_heliostat_data_mappings(
            viable_heliostats_data=viable_heliostats_data,
            heliostats_for_plots=args.heliostats_for_plots,
            sample_limit_kinematics=kinematics_optimization_config["sample_limit"],
            sample_limit_surfaces=surface_optimization_config["sample_limit"]
        )

        create_distributions(
            measured_data_dir=measured_data_dir,
            results_path=results_path,
            device=device,
        )

        # create_deflectometry_surface_for_comparison(
        #     scenario_path=scenario_path_deflectometry,
        #     results_path=results_path,
        #     measured_data_dir=measured_data_dir,
        #     heliostats_for_plots=args.heliostats_for_plots,
        #     number_of_surface_points=surface_optimization_config["number_of_surface_points"],
        #     device=device,
        # )

        loaded = torch.load(results_path, weights_only=False)
        results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)
        target_distribution = results_dict["homogeneous_distribution"]

        # Save heliostat positions.
        heliostat_positions = {}
        with torch.no_grad():
            with h5py.File(scenario_path_ideal) as scenario_file:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=scenario_file,
                    device=device,
                )
                for group in scenario.heliostat_field.heliostat_groups:
                    for name, position in zip(group.names, group.positions):
                        heliostat_positions[name] = position.clone().detach().cpu().tolist()
            results_dict["heliostat_positions"] = heliostat_positions
            torch.save(results_dict, results_path)

        # Logging.
        runtime_log.info(
            f"Number of heliostats: {len(data_mappings['kinematics_reconstruction']['heliostat_data_mapping'])}"
        )
        runtime_log.info(
            f"surface reconstruction: {surface_optimization_config}"
        )
        runtime_log.info(
            f"kinematics reconstruction: {kinematics_optimization_config}"
        )
        runtime_log.info(
            f"aim point optimization: {aim_point_optimization_config}"
        )

        full_field_optimizations(
            scenario_path=scenario_path_ideal,
            results_path=results_path,
            basic_config=basic_config,
            data_mappings=data_mappings,
            surface_config=surface_optimization_config,
            kinematics_config=kinematics_optimization_config,
            aim_point_config=aim_point_optimization_config,
            target_distribution=target_distribution,
            device=device,
        )

        run_info = parse_runtimes("runtime_log.txt")
        loaded = torch.load(results_path, weights_only=False)
        results_dict = cast(dict[str, dict[str, torch.Tensor]], loaded)
        results_dict["run_info"] = run_info
        results_dict["run_info"]["parameters"] = {
            "surface": surface_optimization_config,
            "kinematics": kinematics_optimization_config,
            "aim_points": aim_point_optimization_config
        }
        torch.save(results_dict, results_path)


if __name__ == "__main__":
    runtime_log.info("-----------------")
    main()
    runtime_log.info("\n\n")
