#!/usr/bin/env python
import argparse
import pathlib

import paint.util.paint_mappings as mappings
import pandas as pd
from paint.data.stac_client import StacClient
from paint.util import set_logger_config

set_logger_config()

if __name__ == "__main__":
    """
    This script should be run second to download the required calibration data.

    The data will be downloaded based on the minimum number of calibration measurements required. If a certain heliostat
    does not contain the required number of measurements, its data will not be downloaded.
    """
    # Read in arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save the downloaded data.",
        default="./paint_data",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="Path to the downloaded metadata.",
        default="./metadata",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        help="Name of the metadata file.",
        default="calibration_metadata_all_heliostats.csv",
    )
    parser.add_argument(
        "--minimum_number_of_measurements",
        type=int,
        help="The minimum number of calibration measurements per heliostat required",
        default=80,
    )
    parser.add_argument(
        "--collections",
        type=str,
        help="List of collections to be downloaded.",
        nargs="+",
        choices=[
            mappings.SAVE_DEFLECTOMETRY.lower(),
            mappings.SAVE_CALIBRATION.lower(),
            mappings.SAVE_PROPERTIES.lower(),
        ],
        default=["properties", "calibration"],
    )
    parser.add_argument(
        "--filtered_calibration",
        type=str,
        help="List of calibration items to download.",
        nargs="+",
        choices=[
            mappings.CALIBRATION_RAW_IMAGE_KEY,
            mappings.CALIBRATION_FLUX_IMAGE_KEY,
            mappings.CALIBRATION_FLUX_CENTERED_IMAGE_KEY,
            mappings.CALIBRATION_PROPERTIES_KEY,
            mappings.CALIBRATION_CROPPED_IMAGE_KEY,
        ],
        default=["cropped_image", "calibration_properties"],
    )
    args = parser.parse_args()

    # Create STAC client.
    client = StacClient(output_dir=args.output_dir)

    # Determine viable heliostats, i.e. only those with enough calibration measurements.
    calibration_file = pathlib.Path(args.metadata_dir) / args.metadata_file_name
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
        collections=[mappings.SAVE_CALIBRATION.lower()],
        filtered_calibration_keys=[
            mappings.CALIBRATION_FLUX_IMAGE_KEY,
            mappings.CALIBRATION_FLUX_CENTERED_IMAGE_KEY,
            mappings.CALIBRATION_PROPERTIES_KEY,
        ],
    )
