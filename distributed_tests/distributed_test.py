import argparse
import csv
import logging
import os
import pathlib
import time
from typing import Any
import warnings
import h5py
from matplotlib import pyplot as plt
import torch
import yaml

from artist.flux import bitmap
from artist.optim.aim_point_optimizer import AimPointOptimizer
from artist.optim.loss import KLDivergenceLoss, NormalizedL1
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario

from artist.util import (
    constants,
    indices,
    runtime_log,
    set_logger_config,
    track_runtime,
)
from artist.util.env import (
    get_device,
    setup_distributed_environment,
)

set_logger_config()
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed(7)

log = logging.getLogger(__name__)
"""A logger for the aim point optimizer."""

@track_runtime(runtime_log)
def start_optimize(
    scenario_path: pathlib.Path,
    aim_point_config: dict[str, Any],
    case: str,
    data_dir,
    device: torch.device | None = None,
) -> None:
    """
    Optimize the heliostat field with a combination of surface reconstruction, kinematics reconstruction and aim point optimization.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    results_path : pathlib.Path
        Path to where the results are saved.
    basic_config : dict[str, Any]
        Configuration for number of surface points, number of rays and baseline data.
    data_mappings : dict[str, Any]
        Data mappings including heliostat names and their calibration files, for surface reconstruction, kinematics reconstruction
        and the evaluation plots.
    surface_config : dict[str, Any]
        Configuration parameters for the surface reconstruction.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device=device)

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    optimization_configuration_aim_points = {
        constants.optimization: {
            constants.initial_learning_rate: float(aim_point_config[constants.initial_learning_rate]),
            constants.tolerance: 1e-5,
            constants.max_epoch: int(aim_point_config[constants.max_epoch]),
            constants.batch_size: int(aim_point_config[constants.batch_size]),
            constants.log_step: 1,
            constants.early_stopping_delta: 1e-12,
            constants.early_stopping_patience: 10000,
            constants.early_stopping_window: 10000,
        },
        constants.scheduler: {
            constants.scheduler_type: str(
                aim_point_config[constants.scheduler_type]
            ),
            constants.gamma: float(aim_point_config[constants.gamma]),
            constants.lr_min: float(aim_point_config[constants.lr_min]),
            constants.lr_max: float(aim_point_config[constants.lr_max]),
            constants.step_size_up: int(aim_point_config[constants.step_size_up]),
            constants.reduce_factor: float(
                aim_point_config[constants.reduce_factor]
            ),
            constants.patience: int(aim_point_config[constants.patience]),
            constants.threshold: float(aim_point_config[constants.threshold]),
            constants.cooldown: int(aim_point_config[constants.cooldown]),
        },
        constants.constraints: {
            "lambda_intensity_reward": float(
                aim_point_config["lambda_intensity_reward"]
            ),
            constants.rho_flux_integral: float(
                aim_point_config[constants.rho_flux_integral]
            ),
            constants.rho_local_flux: float(
                aim_point_config[constants.rho_local_flux]
            ),
            constants.max_flux_density: float(
                aim_point_config[constants.max_flux_density]
            ),
        },
    }

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup["device"]

        number_of_surface_points_per_facet = torch.tensor([20, 20], device=device)
        number_of_control_points_per_facet = torch.tensor([4, 4], device=device)

        with h5py.File(scenario_path) as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )

        heliostat_group = scenario.heliostat_field.heliostat_groups[0]
        scenario.set_number_of_rays(number_of_rays=aim_point_config["number_of_rays"])

        bitmap_resolution = torch.tensor(
            aim_point_config["bitmap_resolution"], device=device
        )
        baseline_incident_ray_direction = torch.nn.functional.normalize(
            torch.tensor(
                aim_point_config["baseline_incident_ray_direction"], device=device
            ),
            dim=0,
        )
        baseline_target_area = aim_point_config["baseline_target_area"]
        baseline_dni = aim_point_config["dni"]
        
        ground_truth = torch.load(data_dir / "optimal_flux.pt", weights_only=False, map_location=device)
        
        loss_definition = KLDivergenceLoss()

        aim_point_optimizer = AimPointOptimizer(
            ddp_setup=ddp_setup,
            scenario=scenario,
            optimization_configuration=optimization_configuration_aim_points,
            incident_ray_direction=baseline_incident_ray_direction,
            target_area_index=scenario.solar_tower.target_name_to_index[
                baseline_target_area
            ],
            ground_truth=ground_truth,
            dni=baseline_dni,
            bitmap_resolution=bitmap_resolution,
            device=device,
        )
        aimpoint_optimization_final_loss, intercept_factor_recon = (
            aim_point_optimizer.optimize(loss_definition=loss_definition, device=device)
        )

        motor_positions_optimized = heliostat_group.kinematics.active_motor_positions.detach()

        results = {
            "optimal_flux": ground_truth.detach(),
            "motor_positions_recon": motor_positions_optimized.detach(),
            "intercept_recon": intercept_factor_recon.detach(),
            "target_area_index": scenario.solar_tower.target_name_to_index[
                baseline_target_area
            ],
            "dni": baseline_dni,
            "incident_ray_direction": baseline_incident_ray_direction,
            "bitmap_resolution": bitmap_resolution
        }

        save_dir = pathlib.Path(data_dir)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(results, save_dir / f"optimization_{case}_{int(time.time())}.pt")


def main() -> None:
    """Generate field optimization results and save them."""
    # Locate this script and the repository root (two levels up).
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"
    project_root = script_dir.parent.parent

    def _make_abs(p: str | pathlib.Path) -> pathlib.Path:
        """Resolve a possibly‑relative path relative to the repository root (where YAML paths were written)."""
        p = pathlib.Path(p).expanduser()
        return p if p.is_absolute() else (project_root / p).resolve()

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
    # Resolve directory defaults relative to the repo root.
    data_dir_default = _make_abs(
        config.get("data_dir", "./examples/field_optimizations/scenarios")
    )
    aim_point_optimization_configuration_default = config.get(
        "aim_point_optimization_configuration", {}
    )
    number_heliostats_default = config.get("number_heliostats", "50000")
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the directory for saving the generated scenarios.",
        default=str(data_dir_default),
    )
    parser.add_argument(
        "--aim_point_optimization_configuration",
        help="Config.",
        default=aim_point_optimization_configuration_default,
    )
    parser.add_argument(
        "--number_heliostats",
        help="number_heliostats.",
        default=number_heliostats_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args()
    device = get_device(torch.device(args.device))

    # Convert any CLI‑provided paths (which may still be relative) to absolute paths.
    data_dir = _make_abs(args.data_dir)
    aim_point_optimization_config = args.aim_point_optimization_configuration

    for i in range(10):

        # Define scenario paths and viable heliostat data paths.
        scenario_path_ideal = data_dir / f"ideal_heliostats_{args.number_heliostats}.h5" 

        runtime_log.info(f"Start {i} -----------------")
        start_optimize(
            scenario_path=scenario_path_ideal,
            aim_point_config=aim_point_optimization_config,
            case=i,
            data_dir=data_dir,
            device=device
        )
        runtime_log.info(f"End {i} -----------------")


if __name__ == "__main__":
    main()
