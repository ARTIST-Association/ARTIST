import argparse
import logging
import pathlib
import warnings

import torch
import yaml

from artist.util.environment_setup import get_device

# Assume other functions (find_latest_deflectometry_file, generate_paint_scenario) are unchanged

log = logging.getLogger(__name__)


if __name__ == "__main__":
    """
    Generate two raytracing scenarios: deflectometry and ideal (no deflectometry).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default="./paint_plot_config.yaml",
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
    metadata_dir_default = config.get("metadata_dir", "./metadata")
    metadata_file_name_default = config.get(
        "metadata_file_name", "calibration_metadata_all_heliostats.csv"
    )
    minimum_number_of_measurements_default = config.get(
        "minimum_number_of_measurements", 80
    )
    device_default = config.get("device", "cuda")
    tower_file_name_default = config.get(
        "tower_file_name", "WRI1030197-tower-measurements.json"
    )
    heliostats_default = config.get(
        "heliostats_for_raytracing", ["AA39", "AY26", "BC34"]
    )
    scenarios_dir_default = config.get("scenarios_dir", "./scenarios")

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
        "--tower_file_name",
        type=str,
        help="File name containing the tower data.",
        default=tower_file_name_default,
    )
    parser.add_argument(
        "--heliostats",
        type=str,
        help="List of heliostats to be used in the scenario.",
        nargs="+",
        default=heliostats_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to save the generated scenario.",
        default=scenarios_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    data_dir = pathlib.Path(args.data_dir)
    tower_file = data_dir / args.tower_file_name

    # Generate two scenarios: deflectometry and ideal (no deflectometry).
    deflectometry_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "flux_prediction_deflectometry.h5"
    )
    ideal_scenario_file = pathlib.Path(args.scenarios_dir) / "flux_prediction_ideal.h5"

    for scenario_path, use_def in [
        (deflectometry_scenario_file, True),
        (ideal_scenario_file, False),
    ]:
        if scenario_path.exists():
            print(
                f"Scenario found at {scenario_path}... continue without generating scenario."
            )
        else:
            print(
                f"Scenario not found. Generating a new one at {scenario_path} (use_deflectometry={use_def})..."
            )
            # generate_paint_scenario(
            #     paint_dir=str(data_dir),
            #     scenario_path=scenario_path,
            #     tower_file=str(tower_file),
            #     heliostat_names=args.heliostats,
            #     device=device,
            #     use_deflectometry=use_def,
            # )
