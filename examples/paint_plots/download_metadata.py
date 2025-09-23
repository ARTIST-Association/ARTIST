#!/usr/bin/env python
import argparse
import pathlib
import warnings

import paint.util.paint_mappings as paint_mappings
import yaml
from paint.data.stac_client import StacClient
from paint.util import set_logger_config

set_logger_config()

if __name__ == "__main__":
    """
    This script should be run first to download the required metadata.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    metadata_dir : str
        Path to the directory used for storing the metadata.
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

    metadata_dir_default = config.get("metadata_dir", "./")

    # Read in arguments.
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="Path to the downloaded metadata.",
        default=metadata_dir_default,
    )

    args = parser.parse_args()

    # Create STAC client.
    client = StacClient(output_dir=args.metadata_dir)

    # Download metadata for all heliostats.
    # WARNING: This will take a very long time, but will resume at the same position of the download breaks.
    client.get_heliostat_metadata(collections=[paint_mappings.SAVE_CALIBRATION.lower()])
