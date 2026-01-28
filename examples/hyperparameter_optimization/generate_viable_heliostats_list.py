import argparse
import json
import pathlib
import re
import warnings

import paint.util.paint_mappings as paint_mappings
import yaml


def find_viable_heliostats(
    data_directory: pathlib.Path,
    minimum_number_of_measurements: int,
    kinematic_reconstruction_image_type: str,
    surface_reconstruction_image_type: str,
) -> list[
    tuple[str, list[pathlib.Path], list[pathlib.Path], list[pathlib.Path], pathlib.Path]
]:
    """
    Find heliostats that have at least a minimum number of valid calibration files.

    This function iterates through a data directory to find all heliostats having at least the minimum number of measurements
    calibration files. All paths are collected, and a sorted list of the heliostats is returned containing tuples including
    the heliostat name, path to the calibration file, and path to the flux images for surface and kinematic reconstruction.

    Parameters
    ----------
    data_directory : pathlib.Path
        The path to the data directory.
    minimum_number_of_measurements : int
        The minimum number of calibration files required.
    kinematic_reconstruction_image_type : str
        The type of calibration image to use for the kinematic reconstruction, i.e., ''flux'', or ''flux-centered''.
    surface_reconstruction_image_type : str
        The type of calibration image to use for the surface reconstruction, i.e., ''flux'', or ''flux-centered''.

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path], list[pathlib.Path], pathlib.Path]]
        A list of tuples containing:
        - The heliostat name.
        - A list of valid calibration file paths.
        - A list of flux image file paths for kinematic reconstruction.
        - A list of flux image file paths for surface reconstruction.
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

    for heliostat_directory in sorted(all_heliostats):
        heliostat_name = heliostat_directory.name

        properties_path = (
            heliostat_directory
            / paint_mappings.SAVE_PROPERTIES
            / f"{paint_mappings.HELIOSTAT_PROPERTIES_SAVE_NAME % heliostat_name}"
        )
        calibration_dir = heliostat_directory / paint_mappings.SAVE_CALIBRATION

        if not calibration_dir.exists():
            continue

        valid_calibration_files = []
        flux_images_kinematic_reconstruction = []
        flux_images_surface_reconstruction = []

        for calibration_file_path in sorted(
            calibration_dir.glob(f"*{paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER}")
        ):
            try:
                with calibration_file_path.open("r") as f:
                    calibration_data = json.load(f)
                    focal_spot_data = calibration_data.get(
                        paint_mappings.FOCAL_SPOT_KEY, {}
                    )

                    if paint_mappings.UTIS_KEY in focal_spot_data:
                        # Check for the existence of the corresponding flux image.
                        file_stem = calibration_file_path.stem.removesuffix(
                            json_suffix_to_remove
                        )
                        kinematic_reconstruction_flux_image_path = (
                            calibration_dir
                            / f"{file_stem}-{kinematic_reconstruction_image_type}.png"
                        )
                        surface_reconstruction_flux_image_path = (
                            calibration_dir
                            / f"{file_stem}-{surface_reconstruction_image_type}.png"
                        )

                        if (
                            kinematic_reconstruction_flux_image_path.exists()
                            and surface_reconstruction_flux_image_path.exists()
                        ):
                            valid_calibration_files.append(calibration_file_path)
                            flux_images_kinematic_reconstruction.append(
                                kinematic_reconstruction_flux_image_path
                            )
                            flux_images_surface_reconstruction.append(
                                surface_reconstruction_flux_image_path
                            )
            except Exception as e:
                print(f"Warning: Skipping {calibration_file_path} due to error: {e}")

        if len(valid_calibration_files) >= minimum_number_of_measurements:
            found_heliostats.append(
                (
                    heliostat_name,
                    valid_calibration_files,
                    flux_images_kinematic_reconstruction,
                    flux_images_surface_reconstruction,
                    properties_path,
                )
            )
        print(f"Added heliostat {heliostat_name}")

    return sorted(found_heliostats, key=lambda x: x[0])


if __name__ == "__main__":
    """
    Generate list of viable heliostats for the hyperparameter optimizations.

    This script identifies a list of viable heliostats, i.e., containing a minimum number of valid measurements, for
    the optimization process.

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
    kinematic_reconstruction_image_type : str
        Type of calibration image to use for the kinematic reconstruction, i.e., flux or flux-centered.
    surface_reconstruction_image_type : str
        Type of calibration image to use for the surface reconstruction, i.e., flux or flux-centered.
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
    device_default = config.get("device", "cuda")
    data_dir_default = config.get("data_dir", "./paint_data")
    results_dir_default = config.get(
        "results_dir", "./examples/hyperparameter_optimization/results"
    )
    minimum_number_of_measurements_default = config.get(
        "minimum_number_of_measurements", 8
    )
    kinematic_reconstruction_image_type_default = config.get(
        "kinematic_reconstruction_image_type", "flux"
    )
    surface_reconstruction_image_type_default = config.get(
        "surface_reconstruction_image_type", "flux-centered"
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
        help="Path to the data directory.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to where the results will be saved.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--minimum_number_of_measurements",
        type=int,
        help="Minimum number of calibration measurements per heliostat required.",
        default=minimum_number_of_measurements_default,
    )
    parser.add_argument(
        "--kinematic_reconstruction_image_type",
        type=str,
        help="Type of calibration image to use for the kinematic reconstruction, i.e., flux or flux-centered.",
        choices=["flux", "flux-centered"],
        default=kinematic_reconstruction_image_type_default,
    )
    parser.add_argument(
        "--surface_reconstruction_image_type",
        type=str,
        help="Type of calibration image to use for the surface reconstruction, i.e., flux or flux-centered.",
        choices=["flux", "flux-centered"],
        default=surface_reconstruction_image_type_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    data_dir = pathlib.Path(args.data_dir)

    heliostat_data_list = find_viable_heliostats(
        data_directory=data_dir,
        minimum_number_of_measurements=args.minimum_number_of_measurements,
        kinematic_reconstruction_image_type=args.kinematic_reconstruction_image_type,
        surface_reconstruction_image_type=args.surface_reconstruction_image_type,
    )

    print(f"Selected {len(heliostat_data_list)} heliostats.")

    serializable_data = [
        {
            "name": heliostat_name,
            "calibrations": [
                str(calibration_path) for calibration_path in calibration_paths
            ],
            "kinematic_reconstruction_flux_images": [
                str(flux_path) for flux_path in kinematic_reconstruction_flux_paths
            ],
            "surface_reconstruction_flux_images": [
                str(flux_path) for flux_path in surface_reconstruction_flux_image_path
            ],
            "properties": str(properties_path),
        }
        for heliostat_name, calibration_paths, kinematic_reconstruction_flux_paths, surface_reconstruction_flux_image_path, properties_path in heliostat_data_list
    ]

    results_path = pathlib.Path(args.results_dir) / "viable_heliostats.json"

    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as output_file:
        json.dump(serializable_data, output_file, indent=2)

    print(f"Saved {len(serializable_data)} heliostat entries to {results_path}")
