import copy
import json
import os
import pathlib
from typing import Optional

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

from artist.core.kinematic_calibrator import KinematicCalibrator
from artist.core.loss_functions import AngleLoss, FocalSpotLoss
from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from examples.paint_plots.helpers import filter_valid_heliostat_data, join_safe, load_config, load_heliostat_data
import argparse

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

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
    config_dictionary.data_source: config_dictionary.paint,
    config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
}

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device_used = ddp_setup[config_dictionary.device]

            # Set calibration method and loss function.
        kinematic_calibration_method = config_dictionary.kinematic_calibration_motor_positions
        # Uncomment for calibration with raytracing:
       
        # Uncomment for calibration with motor positions.
        # loss_definition = VectorLoss()

        # Configure the learning rate scheduler. The example scheduler parameter dict includes
        # example parameters for all three possible schedulers.
        scheduler = (
            config_dictionary.exponential
        )  # exponential, cyclic or reduce_on_plateau
        scheduler_parameters = {
            config_dictionary.gamma: 0.999,
            config_dictionary.min: 1e-6,
            config_dictionary.max: 1e-2,
            config_dictionary.step_size_up: 500,
            config_dictionary.reduce_factor: 0.3,
            config_dictionary.patience: 10,
            config_dictionary.threshold: 1e-3,
            config_dictionary.cooldown: 10,
        }

        # Set optimization parameters.
        optimization_configuration = {
            config_dictionary.initial_learning_rate: 0.05,
            config_dictionary.tolerance: 0.000005,
            config_dictionary.max_epoch: 10000,
            config_dictionary.num_log: 100,
            config_dictionary.early_stopping_delta: 1e-6,
            config_dictionary.early_stopping_patience: 100,
            config_dictionary.scheduler: scheduler,
            config_dictionary.scheduler_parameters: scheduler_parameters,
        }

        for centroid in centroids_extracted_by:
            if centroid == config_dictionary.paint_utis:
                scenario = scenario_utis
            elif centroid == config_dictionary.paint_helios:
                scenario = scenario_helios
            else:
                raise ValueError(f"Unknown centroid source: {centroid}")
            loss_definition = AngleLoss()
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
                    centroid_extrected_by=centroid,
                    device=device_used,
                )

                if heliostats_mask_calibration.sum() <= 0:
                    continue
                kinematic_calibrator = KinematicCalibrator(
                    ddp_setup=ddp_setup,
                    scenario=scenario,
                    data=data,
                    optimization_configuration=optimization_configuration,
                    calibration_method=kinematic_calibration_method,
                )

                raw_losses = kinematic_calibrator.calibrate(
                    loss_definition=loss_definition, device=device
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
    
    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            results_dict.setdefault(name, {})["position"] = (
                position.clone().detach().cpu().tolist()
            )

    return results_dict

if __name__ == "__main__":


    config = load_config()

    device = torch.device(config["device"])
    device = get_device(device)
    paint_repository_base_path = config["paint_repository_base_path"]

    paint_plots_base_path = pathlib.Path(config["base_path"])
    scenario_path = join_safe(paint_plots_base_path, config["calibration_scenario_path"])
    heliostat_list_file = join_safe(paint_plots_base_path, config["heliostat_list_path"])
    results_path = join_safe(paint_plots_base_path, config["results_calibration_dict_path"])

    heliostat_data_mapping, heliostat_properties_list = load_heliostat_data(
        paint_repository_base_path, heliostat_list_file
    )

    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )
    scenario_utis = copy.deepcopy(scenario)
    scenario_helios = copy.deepcopy(scenario)

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
    )

    # Inline attach_positions: ensure positions are included before saving.
    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            results_dict.setdefault(name, {})["position"] = (
                position.clone().detach().cpu().tolist()
            )

    torch.save(results_dict, results_path)
    print(f"Calibration results saved to {results_path}")