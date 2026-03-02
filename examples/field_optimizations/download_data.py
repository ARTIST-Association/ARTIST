#!/usr/bin/env python
import argparse
import pathlib
import warnings

import paint.util.paint_mappings as paint_mappings
import pandas as pd
import yaml
from paint.data.stac_client import StacClient
from paint.util import set_logger_config

set_logger_config()

if __name__ == "__main__":
    """
    This script should be run second to download the required calibration, deflectometry, and tower data.

    The data will be downloaded based on the minimum number of calibration measurements required. If a certain heliostat
    does not contain the required number of measurements, its data will not be downloaded.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    data_dir : str
        Path to the directory used for storing the data.
    metadata_root : str
        Path to the root directory where the metadata folder is stored.
    metadata_file_name : str
        Name of the metadata file.
    minimum_number_of_measurements : int
        Minimum number of calibration measurements per heliostat required.
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
    data_dir_default = config.get(
        "data_dir", "./examples/field_optimizations/field_data"
    )
    metadata_root_default = config.get(
        "metadata_root", "./examples/field_optimizations/"
    )
    metadata_file_name_default = config.get(
        "metadata_file_name", "calibration_metadata_all_heliostats.csv"
    )
    minimum_number_of_measurements_default = config.get(
        "minimum_number_of_measurements", 4
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to save the downloaded data.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--metadata_root",
        type=str,
        help="Path to the root containing the metadata folder.",
        default=metadata_root_default,
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        help="Name of the metadata file.",
        default=metadata_file_name_default,
    )
    parser.add_argument(
        "--minimum_number_of_measurements",
        type=int,
        help="The minimum number of calibration measurements per heliostat required",
        default=minimum_number_of_measurements_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    # Create STAC client.
    client = StacClient(output_dir=args.data_dir)

    # Download tower measurements.
    client.get_tower_measurements()

    # Determine viable heliostats i.e., only those with enough calibration measurements.
    calibration_file = (
        pathlib.Path(args.metadata_root) / "metadata" / args.metadata_file_name
    )
    calibration_metadata = pd.read_csv(calibration_file)
    number_of_images_per_heliostat = calibration_metadata.groupby("HeliostatId")[
        "Id"
    ].transform("count")
    viable_heliostats = (
        calibration_metadata[
            number_of_images_per_heliostat >= args.minimum_number_of_measurements
        ]["HeliostatId"]
        .unique()
        .tolist()
    )
    # Download heliostat data.
    client.get_heliostat_data(
        heliostats=viable_heliostats,
        collections=[
            paint_mappings.SAVE_CALIBRATION.lower(),
            paint_mappings.SAVE_DEFLECTOMETRY.lower(),
            paint_mappings.SAVE_PROPERTIES.lower(),
        ],
        filtered_calibration_keys=[
            paint_mappings.CALIBRATION_FLUX_IMAGE_KEY,
            paint_mappings.CALIBRATION_FLUX_CENTERED_IMAGE_KEY,
            paint_mappings.CALIBRATION_PROPERTIES_KEY,
        ],
    )
