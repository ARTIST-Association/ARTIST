import argparse
import json
import pathlib
import warnings
from typing import Any, cast

import h5py
import torch
import yaml

from artist.core import loss_functions
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.regularizers import IdealSurfaceRegularizer, SmoothnessRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment

set_logger_config()
torch.manual_seed(7)
torch.cuda.manual_seed(7)


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


def data_for_flux_plots(
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

        scenario.set_number_of_rays(number_of_rays=30)

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
            bitmaps_for_plots[heliostat] = {
                "measured_flux": measured_fluxes[index],
                "artist_flux": cropped_flux_distributions[index],
                "surface_points": heliostat_group.surface_points[index],
                "surface_normals": heliostat_group.surface_normals[index],
                "canting": heliostat_group.active_canting[index],
                "facet_translations": heliostat_group.active_facet_translations[index],
            }

    return bitmaps_for_plots


def generate_reconstruction_results(
    scenario_path: pathlib.Path,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    hyperparameters: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """
    Perform surface reconstruction in ``ARTIST`` and save results.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to reconstruction scenario.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for each heliostat, containing a list of tuples with the heliostat name, the path to the calibration
        properties file, and the path to the flux images.
    hyperparameters : dict[str, Any]
        Optimized hyperparameters.
    device : torch.device | None
        Device used for optimization and tensor allocations.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from heliostat name to per-centroid loss arrays and, later, positions.
    """
    device = get_device(device=device)

    loss_dict: dict = {}

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        number_of_surface_points_per_facet = torch.tensor(
            [
                hyperparameters["number_of_surface_points"],
                hyperparameters["number_of_surface_points"],
            ],
            device=device,
        )

        number_of_control_points_per_facet = torch.tensor(
            [
                hyperparameters["number_of_control_points"],
                hyperparameters["number_of_control_points"],
            ],
            device=device,
        )

        with h5py.File(
            pathlib.Path(scenario_path.parent) / "deflectometry_scenario_surface.h5",
            "r",
        ) as deflectometry_scenario_file:
            scenario_deflectometry = Scenario.load_scenario_from_hdf5(
                scenario_file=deflectometry_scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                device=device,
            )

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )

        selected_heliostats = [
            "AC38",
            "BD38",
            "AE34",
            "BG65",
            "AK26",
            "AK17",
            "BA43",
            "AZ28",
            "AP51",
            "AP35",
        ]

        scenario.set_number_of_rays(
            number_of_rays=int(hyperparameters["number_of_rays"])
        )

        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            heliostat_group.nurbs_degrees = torch.tensor(
                [hyperparameters["nurbs_degree"], hyperparameters["nurbs_degree"]],
                device=device,
            )

        optimizer_dict = {
            config_dictionary.initial_learning_rate: hyperparameters[
                "initial_learning_rate"
            ],
            config_dictionary.tolerance: 0,
            config_dictionary.max_epoch: 3,
            config_dictionary.batch_size: hyperparameters["sample_limit"] * 2,
            config_dictionary.log_step: 1,
            config_dictionary.early_stopping_delta: 1e-4,
            config_dictionary.early_stopping_patience: 150,
            config_dictionary.early_stopping_window: 150,
        }
        scheduler_dict = {
            config_dictionary.scheduler_type: hyperparameters["scheduler"],
            config_dictionary.gamma: hyperparameters["gamma"],
            config_dictionary.min: hyperparameters["min_learning_rate"],
            config_dictionary.max: hyperparameters["max_learning_rate"],
            config_dictionary.step_size_up: hyperparameters["step_size_up"],
            config_dictionary.reduce_factor: hyperparameters["reduce_factor"],
            config_dictionary.patience: hyperparameters["patience"],
            config_dictionary.threshold: hyperparameters["threshold"],
            config_dictionary.cooldown: hyperparameters["cooldown"],
        }
        ideal_surface_regularizer = IdealSurfaceRegularizer(reduction_dimensions=(1,))
        smoothness_regularizer = SmoothnessRegularizer(reduction_dimensions=(1,))
        regularizers = [
            ideal_surface_regularizer,
            smoothness_regularizer,
        ]
        constraint_dict = {
            config_dictionary.regularizers: regularizers,
            config_dictionary.initial_lambda_energy: 0.1,
            config_dictionary.rho_energy: 1.0,
            config_dictionary.energy_tolerance: 0.01,
            config_dictionary.weight_smoothness: 0.005,
            config_dictionary.weight_ideal_surface: 0.005,
        }
        optimization_configuration = {
            config_dictionary.optimization: optimizer_dict,
            config_dictionary.scheduler: scheduler_dict,
            config_dictionary.constraints: constraint_dict,
        }
        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: PaintCalibrationDataParser(
                sample_limit=hyperparameters["sample_limit"],
            ),
            config_dictionary.heliostat_data_mapping: [
                mapping
                for mapping in heliostat_data_mapping
                if mapping[0] in selected_heliostats
            ],
        }

        data_plot: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: PaintCalibrationDataParser(
                sample_limit=1,
            ),
            config_dictionary.heliostat_data_mapping: [
                mapping
                for mapping in heliostat_data_mapping
                if mapping[0] in selected_heliostats
            ],
        }

        deflectometry_data = data_for_flux_plots(
            scenario=scenario_deflectometry,
            ddp_setup=ddp_setup,
            heliostat_data=data_plot,
            device=device,
        )

        surface_reconstructor = SurfaceReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            number_of_surface_points=number_of_surface_points_per_facet,
            bitmap_resolution=torch.tensor([256, 256], device=device),
            device=device,
        )

        loss_definition = loss_functions.KLDivergenceLoss()

        plot_data_before = data_for_flux_plots(
            scenario=scenario,
            ddp_setup=ddp_setup,
            heliostat_data=data_plot,
            device=device,
        )

        per_heliostat_losses = surface_reconstructor.reconstruct_surfaces(
            loss_definition=loss_definition, device=device
        )

        plot_data_after = data_for_flux_plots(
            scenario=scenario,
            ddp_setup=ddp_setup,
            heliostat_data=data_plot,
            device=device,
        )

        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            for index, name in enumerate(heliostat_group.names):
                loss_dict.setdefault(name, {})
                loss_dict[name]["loss"] = per_heliostat_losses[index].detach().item()

        flux_data = merge_data(plot_data_before, plot_data_after)

    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            loss_dict[name]["position"] = position.clone().detach().cpu().tolist()

    results = {
        "loss": loss_dict,
        "flux": flux_data,
        "deflectometry": deflectometry_data,
    }

    return results


if __name__ == "__main__":
    """
    Generate results with the optimized parameters.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_dir : str
        Path to the data directory.
    heliostat_for_reconstruction : dict[str, list[int]]
        The heliostat and its calibration numbers.
    results_dir : str
        Path to where the results will be saved.
    scenarios_dir : str
        Path to the directory containing the scenarios.
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
    data_dir_default = config.get("data_dir", "./paint_data")
    device_default = config.get("device", "cuda")
    scenarios_dir_default = config.get(
        "scenarios_dir", "./examples/hyperparameter_optimization/scenarios"
    )
    results_dir_default = config.get(
        "results_dir", "./examples/hyperparameter_optimization/results"
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to downloaded paint data.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to directory containing the generated scenarios.",
        default=scenarios_dir_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to save the results.",
        default=results_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)
    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    results_dir = pathlib.Path(args.results_dir)

    # Define scenario path.
    scenario_file = pathlib.Path(args.scenarios_dir) / "ideal_scenario_surface.h5"
    if not scenario_file.exists():
        raise FileNotFoundError(
            f"The reconstruction scenario located at {scenario_file} could not be found! Please run the ``generate_scenario.py`` to generate this scenario, or adjust the file path and try again."
        )

    viable_heliostats_data = (
        pathlib.Path(args.results_dir) / "viable_heliostats_surface.json"
    )
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``generate_viable_heliostat_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Load viable heliostats data.
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)

    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["surface_reconstruction_flux_images"]],
        )
        for item in viable_heliostats
    ]

    with open(results_dir / "hpo_results_surface.json", "r") as file:
        hyperparameters = json.load(file)

    reconstruction_results = generate_reconstruction_results(
        scenario_path=scenario_file,
        heliostat_data_mapping=heliostat_data_mapping,
        hyperparameters=hyperparameters,
        device=device,
    )

    results_path = pathlib.Path(args.results_dir) / "surface_reconstruction_results.pt"
    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(reconstruction_results, results_path)
    print(f"Reconstruction results saved to {results_path}")
