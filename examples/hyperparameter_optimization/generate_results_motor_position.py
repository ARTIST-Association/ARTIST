import argparse
import json
import pathlib
import warnings
from typing import Any

import h5py
import torch
import yaml

from artist.core import loss_functions
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment

set_logger_config()
torch.manual_seed(7)
torch.cuda.manual_seed(7)


def data_for_flux_plots(
    scenario: Scenario,
    incident_ray_direction: torch.Tensor,
    target_area_index: int,
    dni: float,
    id: str,
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
        Dictionary containing kinematics data per heliostat.
    """
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
        (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
            scenario.index_mapping(
                heliostat_group=heliostat_group,
                single_incident_ray_direction=incident_ray_direction,
                single_target_area_index=target_area_index,
                device=device,
            )
        )

        # Activate heliostats.
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Align Heliostats.
        if id == "before":
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.target_areas.centers[target_area_mask],
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
        (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
            scenario.index_mapping(
                heliostat_group=heliostat_group,
                single_incident_ray_direction=incident_ray_direction,
                single_target_area_index=target_area_index,
                device=device,
            )
        )

        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=True,
            batch_size=100,
            bitmap_resolution=bitmap_resolution,
            dni=dni,
        )

        # Perform heliostat-based ray tracing.
        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
            bitmaps_per_heliostat=bitmaps_per_heliostat,
            target_area_mask=target_area_mask,
            device=device,
        )[target_area_index]

        total_flux += flux_distribution_on_target

    return total_flux


def generate_reconstruction_results(
    scenario_path: pathlib.Path,
    incident_ray_direction: torch.Tensor,
    target_area_index: int,
    target_distribution: torch.Tensor,
    dni: float,
    hyperparameters: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """
    Perform kinematics reconstruction in ``ARTIST`` and save results.

    This function performs the kinematics reconstruction in ``ARTIST`` and saves the results. Reconstruction is compared when using the
    focal spot centroids extracted from HELIOS and the focal spot centroids extracted from UTIS. The results are saved
    for plotting later.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to reconstruction scenario.
    incident_ray_direction : torch.Tensor
        The incident ray direction during the optimization.
        Tensor of shape [4].
    target_area_index : int
        The index of the target used for the optimization.
    target_distribution : torch.Tensor
        The desired focal spot or distribution.
        Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    dni : float
        Direct normal irradiance in W/m^2.
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

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                device=device,
            )

        scenario.set_number_of_rays(number_of_rays=3)
        optimizer_dict = {
            config_dictionary.initial_learning_rate: hyperparameters[
                "initial_learning_rate"
            ],
            config_dictionary.tolerance: 0,
            config_dictionary.max_epoch: 3,
            config_dictionary.batch_size: 100,
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
        constraint_dict = {
            config_dictionary.rho_energy: 1.0,
            config_dictionary.max_flux_density: 300,
            config_dictionary.rho_pixel: 1.0,
            config_dictionary.lambda_lr: 0.1,
        }
        optimization_configuration = {
            config_dictionary.optimization: optimizer_dict,
            config_dictionary.scheduler: scheduler_dict,
            config_dictionary.constraints: constraint_dict,
        }

        motor_positions_optimizer = MotorPositionsOptimizer(
            ddp_setup=ddp_setup,
            scenario=scenario,
            optimization_configuration=optimization_configuration,
            incident_ray_direction=incident_ray_direction,
            target_area_index=target_area_index,
            ground_truth=target_distribution,
            bitmap_resolution=torch.tensor([256, 256]),
            dni=dni,
            device=device,
        )

        flux_before = data_for_flux_plots(
            scenario=scenario,
            incident_ray_direction=incident_ray_direction,
            target_area_index=target_area_index,
            dni=dni,
            id="before",
            device=device,
        )

        loss = motor_positions_optimizer.optimize(
            loss_definition=loss_functions.KLDivergenceLoss(), device=device
        )

        flux_after = data_for_flux_plots(
            scenario=scenario,
            incident_ray_direction=incident_ray_direction,
            target_area_index=target_area_index,
            dni=dni,
            id="after",
            device=device,
        )

        results = {
            "flux_before": flux_before,
            "flux_after": flux_after,
            "target_distribution": target_distribution,
            "loss": loss,
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
    scenario_file = (
        pathlib.Path(args.scenarios_dir) / "deflectometry_scenario_surface.h5"
    )
    if not scenario_file.exists():
        raise FileNotFoundError(
            f"The optimization scenario located at {scenario_file} could not be found! Please run the ``generate_scenario.py`` to generate this scenario, or adjust the file path and try again."
        )

    # DNI W/m^2.
    dni = 850
    # Incident ray direction.
    incident_ray_direction = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        - torch.tensor([0.0, 0.0, 1.0, 1.0], device=device),
        dim=0,
    )
    # Receiver.
    target_area_index = 1
    # Target distribution.
    e_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=120, device=device
    )
    u_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=120, device=device
    )
    eu_trapezoid = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

    target_distribution = (eu_trapezoid / eu_trapezoid.sum()) * 2810000.00

    with open(results_dir / "hpo_results_motor_positions.json", "r") as file:
        hyperparameters = json.load(file)

    optimization_results = generate_reconstruction_results(
        scenario_path=scenario_file,
        incident_ray_direction=incident_ray_direction,
        target_area_index=target_area_index,
        target_distribution=target_distribution,
        dni=dni,
        hyperparameters=hyperparameters,
        device=device,
    )

    results_path = (
        pathlib.Path(args.results_dir) / "motor_position_optimization_results.pt"
    )
    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(optimization_results, results_path)
    print(f"Reconstruction results saved to {results_path}")
