import argparse
import copy
import pathlib
from typing import Optional

import h5py
import torch

from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from examples.paint_plots import paint_plot_helpers
from examples.paint_plots.paint_plot_helpers import (
    filter_valid_heliostat_data,
    join_safe,
    load_config,
)

torch.manual_seed(7)
torch.cuda.manual_seed(7)

FIGSIZE = (6, 4)
LEGEND_FONTSIZE = 8


# Set up logger.
set_logger_config()

# ---------------
# helper functions
# ---------------


def run_calibration(
    scenario_utis: Scenario,
    scenario_helios: Scenario,
    valid_heliostat_data_mapping: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ],
    device: Optional[torch.device],
    centroids_extracted_by: Optional[list[str]] = None,
    # optimizer settings moved from inside function to parameters:
    use_ray_tracing: bool = False,
    tolerance: float = 0.05,
    max_epoch: int = 1000,
    initial_learning_rate: float = 0.001,
) -> dict:
    """Run distributed calibration for all groups and selected centroid sources.

    Parameters
    ----------
    scenario_utis : Scenario
        Scenario object used when centroids are extracted by UTIS.
    scenario_helios : Scenario
        Scenario object used when centroids are extracted by HeliOS.
    valid_heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        For each heliostat, (name, calibration_property_paths, flux_image_paths) after filtering.
    device : torch.device | None
        Device used for optimization and tensor allocations.
    centroids_extracted_by : list[str] | None, default=None
        Centroid sources to run. Defaults to [paint_utis, paint_helios].
    use_ray_tracing : bool, default=False
        If True, disables motor positions calibration (ray-tracing-based optimization expected).
    tolerance : float, default=0.05
        Target tolerance for the optimizer.
    max_epoch : int, default=1000
        Maximum number of optimization epochs.
    initial_learning_rate : float, default=0.001
        Learning rate for the optimizer.

    Returns
    -------
    dict
        Mapping from heliostat name to per-centroid loss arrays and, later, positions.
    """
    centroids_extracted_by = centroids_extracted_by or [
        config_dictionary.paint_utis,
        config_dictionary.paint_helios,
    ]

    number_of_heliostat_groups = len(scenario_utis.heliostat_field.heliostat_groups)
    # Validate that UTIS and HeliOS scenarios have matching topology.
    number_of_heliostat_groups_utis = len(
        scenario_utis.heliostat_field.heliostat_groups
    )
    number_of_heliostat_groups_helios = len(
        scenario_helios.heliostat_field.heliostat_groups
    )
    if number_of_heliostat_groups_utis != number_of_heliostat_groups_helios:
        raise ValueError(
            f"Mismatch in number of heliostat groups: UTIS={number_of_heliostat_groups_utis}, HeliOS={number_of_heliostat_groups_helios}. You have to load the same scenario twice!"
        )
    results_dict: dict = {}

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as (
        device_ctx,
        is_distributed,
        is_nested,
        rank,
        world_size,
        process_subgroup,
        groups_to_ranks_mapping,
        heliostat_group_rank,
        heliostat_group_world_size,
    ):
        device_used = device_ctx

        for centroid in centroids_extracted_by:
            if centroid == config_dictionary.paint_utis:
                scenario = scenario_utis
            elif centroid == config_dictionary.paint_helios:
                scenario = scenario_helios
            else:
                raise ValueError(f"Unknown centroid source: {centroid}")
            for heliostat_group in scenario.heliostat_field.heliostat_groups:
                (
                    _focal_spots_calibration,
                    _incident_ray_directions_calibration,
                    _motor_positions_calibration,
                    heliostats_mask_calibration,
                    _target_area_mask_calibration,
                ) = paint_loader.extract_paint_calibration_properties_data(
                    heliostat_calibration_mapping=[
                        (heliostat_name, calibration_properties_paths)
                        for heliostat_name, calibration_properties_paths, _ in valid_heliostat_data_mapping
                        if heliostat_name in heliostat_group.names
                    ],
                    heliostat_names=heliostat_group.names,
                    target_area_names=scenario.target_areas.names,
                    power_plant_position=scenario.power_plant_position,
                    centroid_extraction_method=centroid,
                    device=device_used,
                )

                if heliostats_mask_calibration.sum() <= 0:
                    continue

                # Disable motor positions if using ray tracing mode (data unused here).
                if use_ray_tracing:
                    _motor_positions_calibration = None

                kinematic_optimizer = KinematicOptimizer(
                    scenario=scenario,
                    heliostat_group=heliostat_group,
                    heliostat_data_mapping=valid_heliostat_data_mapping,
                    calibration_method="motor_positions",
                    tolerance=tolerance,
                    num_log_epochs=10,
                    max_epoch=max_epoch,
                    loss_type="l1",
                    loss_reduction="none",
                    loss_return_value="angular",
                )

                raw_losses = kinematic_optimizer.optimize(
                    device=device_used,
                )
                if raw_losses is None:
                    continue
                losses = raw_losses

                for index, name in enumerate(heliostat_group.names):
                    start = heliostats_mask_calibration[:index].sum()
                    end = heliostats_mask_calibration[: index + 1].sum()
                    per_heliostat_losses = losses[start:end].detach().cpu().numpy()
                    if name not in results_dict:
                        results_dict[name] = {}
                    results_dict[name][centroid] = per_heliostat_losses

    return results_dict


if __name__ == "__main__":
    config = load_config()

    paint_directory = pathlib.Path(config["paint_repository_base_path"])
    tower_file = join_safe(paint_directory, config["paint_tower_file"])

    paint_plot_base_path = pathlib.Path(config["base_path"])
    heliostat_list_file = join_safe(paint_plot_base_path, config["heliostat_list_path"])
    scenario_path = join_safe(paint_plot_base_path, config["calibration_scenario_path"])
    results_path = join_safe(paint_plot_base_path, config["result_dict_path"])
    save_plot_path = join_safe(paint_plot_base_path, config["result_plot_path"])

    # Parse non-path settings from CLI, not from config file
    parser = argparse.ArgumentParser(description="ARTIST paint calibration example")
    parser.add_argument(
        "--use-ray-tracing",
        action="store_true",
        default=False,
        help="Enable ray tracing optimization mode",
    )
    parser.add_argument(
        "--tolerance", type=float, default=None, help="Target tolerance for optimizer"
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Maximum number of optimization epochs",
    )
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=None,
        help="Learning rate for optimizer",
    )
    args = parser.parse_args()

    device = get_device()

    use_ray_tracing = args.use_ray_tracing
    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else (0.035 if use_ray_tracing else 0.05)
    )
    max_epoch = (
        args.max_epoch
        if args.max_epoch is not None
        else (600 if use_ray_tracing else 1000)
    )
    initial_learning_rate = (
        args.initial_learning_rate
        if args.initial_learning_rate is not None
        else (0.005 if use_ray_tracing else 0.003)
    )

    # Run calibration.
    print("No existing results found. Running main()...")
    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )

    # Two scenarios are created to train twice. Once with UTIS centroids and once with HeliOS centroids starting from untrained heliostat states.
    scenario_utis = copy.deepcopy(scenario)
    scenario_helios = copy.deepcopy(scenario)

    heliostat_data_mapping, heliostat_properties_list = (
        paint_plot_helpers.load_heliostat_data(paint_directory, heliostat_list_file)
    )

    valid_heliostat_data_mapping = filter_valid_heliostat_data(heliostat_data_mapping)

    results_dict = run_calibration(
        scenario_utis=scenario_utis,
        scenario_helios=scenario_helios,
        valid_heliostat_data_mapping=valid_heliostat_data_mapping,
        device=device,
        centroids_extracted_by=[
            config_dictionary.paint_utis,
            config_dictionary.paint_helios,
        ],
        use_ray_tracing=use_ray_tracing,
        tolerance=tolerance,
        max_epoch=max_epoch,
        initial_learning_rate=initial_learning_rate,
    )

    # Inline attach_positions: ensure positions are included before saving.
    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            results_dict.setdefault(name, {})["position"] = (
                position.clone().detach().cpu().tolist()
            )

    torch.save(results_dict, results_path)
    print(f"Calibration results saved to {results_path}")
