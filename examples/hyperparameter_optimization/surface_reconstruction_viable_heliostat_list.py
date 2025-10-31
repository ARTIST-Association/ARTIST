import argparse
import json
import pathlib
import warnings

import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.util.environment_setup import get_device


def find_heliostat_files(
    data_directory: pathlib.Path,
    heliostat_calibrations: dict[str, list[int]],
    calibration_image_type: str,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path], pathlib.Path]]:
    """
    Find the heliostats and its calibration data.

    This function searches for the requested heliostat, if it is found the paths are collected.
    The result contains a tuple including the heliostat name, path to the calibration files, and path to the flux images.
    A calibration JSON is considered valid when its focal-spot section contains both centroid extracted by HeliOS and
    UTIS.

    Parameters
    ----------
    data_directory : pathlib.Path
        The path to the data directory.
    heliostat_calibrations : dict[str, list[int]]
        The selected heliostat and its calibration numbers.
    calibration_image_type : str
        The type of calibration image to use, i.e., ''flux'', or ''flux-centered''.

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path], pathlib.Path]]
        A list of tuples containing:
        - The heliostat name.
        - A list of valid calibration file paths.
        - A list of flux image file paths.
        - The associated heliostat properties path.
    """
    found_heliostats = []

    json_suffix_to_remove = (
        paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER.removesuffix(".json")
    )
    
    for heliostat_name in list(heliostat_calibrations.keys()):
    #heliostat_name = list(heliostat_calibrations.keys())

        heliostat_dir = data_directory / heliostat_name

        if not heliostat_dir.is_dir():
            heliostat_dir = data_directory / "AA39"
            raise ValueError(f"No data found for {heliostat_name}.")

        properties_path = (
            heliostat_dir
            / paint_mappings.SAVE_PROPERTIES
            / f"{paint_mappings.HELIOSTAT_PROPERTIES_SAVE_NAME % heliostat_name}"
        )
        calibration_dir = heliostat_dir / paint_mappings.SAVE_CALIBRATION

        valid_calibration_files = []
        flux_images = []

        calibration_numbers = set(map(str, heliostat_calibrations[heliostat_name]))
        matching_files = [
            file
            for file in calibration_dir.iterdir()
            if file.is_file()
            and any(number in file.name for number in calibration_numbers)
            and paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER in file.name
        ]

        for calibration_file_path in matching_files:
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
                            calibration_dir / f"{file_stem}-{calibration_image_type}.png"
                        )

                        if flux_image_path.exists():
                            valid_calibration_files.append(calibration_file_path)
                            flux_images.append(flux_image_path)
            except Exception as e:
                print(f"Warning: Skipping {calibration_file_path} due to error: {e}")

        found_heliostats.append(
            (
                heliostat_name,
                valid_calibration_files,
                flux_images,
                properties_path,
            )
        )

    return sorted(found_heliostats, key=lambda x: x[0])


if __name__ == "__main__":
    """
    Generate list of viable heliostats for the reconstruction.

    This script searches for the selected heliostat and its calibration data.

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
    heliostat_for_reconstruction : dict[str, list[int]]
        The heliostat and its calibration numbers.
    calibration_image_type : str
        Type of calibration image to use, either flux or flux-centered.
    """
    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "hpo_config.yaml"

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
    results_dir_default = config.get("results_dir", "./examples/hyperparameter_optimization/results")
    heliostat_for_reconstruction_default = config.get(
        "heliostat_for_reconstruction", {"AA39": [244862, 270398, 246213, 258959]}
    )
    calibration_image_type_default = config.get(
        "calibration_image_type", "flux-centered"
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
        "--results_dir",
        type=str,
        help="Path to save the results.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--heliostat_for_reconstruction",
        type=str,
        help="The heliostat and its calibration numbers to be reconstructed.",
        nargs="+",
        default=heliostat_for_reconstruction_default,
    )
    parser.add_argument(
        "--calibration_image_type",
        type=str,
        help="Type of calibration image to use, i.e., flux or flux-centered.",
        choices=["flux", "flux-centered"],
        default=calibration_image_type_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)

    heliostat_data_list = find_heliostat_files(
        data_directory=data_dir,
        heliostat_calibrations=args.heliostat_for_reconstruction,
        calibration_image_type=args.calibration_image_type,
    )

    print(f"Selected {len(heliostat_data_list)} heliostat:")
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

    results_path = (
        pathlib.Path(args.results_dir) / "surface_reconstruction_viable_heliostats.json"
    )
    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as output_file:
        json.dump(serializable_data, output_file, indent=2)
    print(f"Saved {len(serializable_data)} heliostat entry to {results_path}")
