import argparse
import json
import pathlib
import re
import warnings

import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.util.environment_setup import get_device


def find_viable_heliostats(
    data_directory: pathlib.Path,
    minimum_number_of_measurements: int,
    maximum_number_of_heliostats: int,
    excluded_heliostats: set[str],
    calibration_image_type: str,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path], pathlib.Path]]:
    """
    Find heliostats that have at least a minimum number of valid calibration files.

    This function iterates through a data directory to find all heliostat calibration files that contain a valid focal
    spot key. In this case the paths are collected and a sorted list of up to the maximum number of heliostats is
    returned containing tuples including the heliostat name, path to the calibration file, and path to the flux image.
    A calibration JSON is considered valid when its focal-spot section contains both centroid extracted by HeliOS and
    UTIS.

    Parameters
    ----------
    data_directory : pathlib.Path
        The path to the data directory.
    minimum_number_of_measurements : int
        The minimum number of calibration files required.
    maximum_number_of_heliostats : int
        The maximum number of heliostats to include in the result.
    excluded_heliostats : set[str]
        The set of heliostats to exclude.
    calibration_image_type : str
        The type of calibration image to use, i.e. ''flux'', or ''flux-centered''.

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path], pathlib.Path]]
        A list of tuples containing:
        - The heliostat name.
        - A list of valid calibration file paths.
        - A list of flux image file paths.
        - The associated heliostat properties path.
    """
    heliostat_name_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}$")
    found_heliostats = []

    json_suffix_to_remove = (
        paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER.removesuffix(".json")
    )

    all_heliostats = (
        d
        for d in data_directory.iterdir()
        if d.is_dir() and heliostat_name_pattern.match(d.name)
    )

    for heliostat_dir in sorted(all_heliostats):
        heliostat_name = heliostat_dir.name

        # Add the exclusion check here
        if heliostat_name in excluded_heliostats:
            print(f"Skipping excluded heliostat: {heliostat_name}")
            continue

        properties_path = (
            heliostat_dir
            / paint_mappings.SAVE_PROPERTIES
            / f"{paint_mappings.HELIOSTAT_PROPERTIES_SAVE_NAME % heliostat_name}"
        )
        calibration_dir = heliostat_dir / paint_mappings.SAVE_CALIBRATION

        if not calibration_dir.exists():
            continue

        valid_calibration_files = []
        flux_images = []

        for calibration_file_path in sorted(
            calibration_dir.glob(f"*{paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER}")
        ):
            try:
                with calibration_file_path.open("r") as f:
                    calibration_data = json.load(f)
                    focal_spot_data = calibration_data.get(
                        paint_mappings.FOCAL_SPOT_KEY, {}
                    )

                    if (
                        paint_mappings.HELIOS_KEY in focal_spot_data
                        and paint_mappings.UTIS_KEY in focal_spot_data
                    ):
                        # Check for the existence of the corresponding flux image.
                        file_stem = calibration_file_path.stem.removesuffix(
                            json_suffix_to_remove
                        )
                        flux_image_path = (
                            calibration_dir
                            / f"{file_stem}-{calibration_image_type}.png"
                        )

                        if flux_image_path.exists():
                            valid_calibration_files.append(calibration_file_path)
                            flux_images.append(flux_image_path)
            except Exception as e:
                print(f"Warning: Skipping {calibration_file_path} due to error: {e}")

        if len(valid_calibration_files) >= minimum_number_of_measurements:
            found_heliostats.append(
                (
                    heliostat_name,
                    valid_calibration_files[:minimum_number_of_measurements],
                    flux_images[:minimum_number_of_measurements],
                    properties_path,
                )
            )
            print(
                f"Added heliostat {heliostat_name}. Found {len(found_heliostats)} so far."
            )

        if len(found_heliostats) >= maximum_number_of_heliostats:
            break

    return sorted(found_heliostats, key=lambda x: x[0])


if __name__ == "__main__":
    """
    Generate list of viable heliostats for calibration.

    This script identifies a list of viable heliostats, i.e. containing a minimum number of valid measurements, for
    the calibration process.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_dir : str
        Path to the data directory.
    results_dir : str
        Path to where the results will be saved.
    minimum_number_of_measurements : int
        Minimum number of calibration measurements per heliostat required.
    maximum_number_of_heliostats : int
        Maximum number of heliostats to include.
    excluded_heliostats : list[str]
        List of heliostats to exclude.
    calibration_image_type : str
        Type of calibration image to use, either flux or flux-centered.
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
    device_default = config.get("device", "cuda")
    results_dir_default = config.get("results_dir", "./results")
    minimum_number_of_measurements_default = config.get(
        "minimum_number_of_measurements", 80
    )
    maximum_number_of_heliostats_default = config.get(
        "maximum_number_of_heliostats_for_calibration", 100
    )
    excluded_heliostats_default = config.get(
        "excluded_heliostats_for_calibration", ["AA39"]
    )
    calibration_image_type_default = config.get("calibration_image_type", "flux")

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
        "--results_dir",
        type=str,
        help="Path to save the results.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--minimum_number_of_measurements",
        type=int,
        help="The minimum number of calibration measurements per heliostat required",
        default=minimum_number_of_measurements_default,
    )
    parser.add_argument(
        "--maximum_number_of_heliostats",
        type=int,
        help="Maximum number of heliostats to include.",
        default=maximum_number_of_heliostats_default,
    )
    parser.add_argument(
        "--excluded_heliostats",
        type=str,
        help="Heliostat names to exclude.",
        nargs="+",
        default=excluded_heliostats_default,
    )
    parser.add_argument(
        "--calibration_image_type",
        type=str,
        help="Type of calibration image to use, i.e. flux or flux-centered",
        choices=["flux", "flux-centered"],
        default=calibration_image_type_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)

    excluded_heliostats: set[str] = set(args.excluded_heliostats)

    heliostat_data_list = find_viable_heliostats(
        data_directory=data_dir,
        minimum_number_of_measurements=args.minimum_number_of_measurements,
        maximum_number_of_heliostats=args.maximum_number_of_heliostats,
        excluded_heliostats=excluded_heliostats,
        calibration_image_type=args.calibration_image_type,
    )

    print(f"Selected {len(heliostat_data_list)} heliostats:")
    for (
        heliostat_name,
        calibration_paths,
        flux_paths,
        _,
    ) in heliostat_data_list:
        print(
            f"- {heliostat_name}: {len(calibration_paths)} calibrations, {len(flux_paths)} flux images ({args.calibration_image_type})"
        )

    serializable_data = [
        {
            "name": heliostat_name,
            "calibrations": [
                str(calibration_path) for calibration_path in calibration_paths
            ],
            "flux_images": [str(flux_path) for flux_path in flux_paths],
            "properties": str(properties_path),
        }
        for heliostat_name, calibration_paths, flux_paths, properties_path in heliostat_data_list
    ]

    results_path = pathlib.Path(args.results_dir) / "viable_heliostats.json"
    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as output_file:
        json.dump(serializable_data, output_file, indent=2)
    print(f"Saved {len(serializable_data)} heliostat entries to {results_path}")
