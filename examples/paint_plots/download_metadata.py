#!/usr/bin/env python
import argparse

import paint.util.paint_mappings as paint_mappings
from paint.data.stac_client import StacClient
from paint.util import set_logger_config

set_logger_config()

if __name__ == "__main__":
    """
    This script should be run first to download the required metadata.
    """
    # Read in arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save the downloaded metadata.",
        default="./",
    )
    args = parser.parse_args()

    # Create STAC client.
    client = StacClient(output_dir=args.output_dir)

    # Download metadata for all heliostats.
    # WARNING: This will take a very long time, but will resume at the same position of the download breaks.
    client.get_heliostat_metadata(collections=[paint_mappings.SAVE_CALIBRATION.lower()])
