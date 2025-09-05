import json
import os
import pathlib
import re

from artist.util import config_dictionary
from examples.paint_plots.helpers import join_safe, load_config
import argparse


def find_heliostats_with_minimum_calibrations(
    paint_directory: str | pathlib.Path,
    minimum_calibration_files: int = 100,
    maximum_heliostats: int = 10,
    flux_file_suffix: str = "flux",
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """
    Find heliostats that have at least a minimum number of valid calibration files.

    Scan a ''PAINT'' directory for heliostat subdirectories that match the naming pattern, validate calibration JSONs
    for required focal-spot keys, and collect matching flux image paths for a given suffix. The function returns a
    sorted list of up to `maximum_heliostats` tuples of the form (heliostat_name, calibration_paths, flux_image_paths).
    A calibration JSON is considered valid when its focal-spot section contains both
    `config_dictionary.paint_helios` and `config_dictionary.paint_utis`; only the first `minimum_files` valid
    calibrations per heliostat are used and missing or unreadable JSON files are skipped with a warning.

    Parameters
    ----------
    paint_directory : str or pathlib.Path
        The path to the base ''PAINT'' directory.
    minimum_calibration_files : int
        The minimum number of calibration files required.
    maximum_heliostats : int
        The maximum number of heliostats to include in the result.
    flux_file_suffix : str
        The suffix to use for flux image filenames (e.g., ''flux'', ''flux-centered'').

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        A list of tuples containing:
        - The heliostat name.
        - A list of valid calibration file paths.
        - A list of flux image file paths.
    """
    paint_directory = pathlib.Path(paint_directory)
    heliostat_name_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}$")
    heliostat_data_list = []

    for subdirectory in paint_directory.iterdir():
        if not subdirectory.is_dir():
            continue

        heliostat_name = subdirectory.name
        if not heliostat_name_pattern.match(heliostat_name):
            continue

        calibration_directory = (
            subdirectory / config_dictionary.paint_calibration_folder_name
        )
        if not calibration_directory.exists():
            continue

        calibration_file_paths = sorted(
            calibration_directory.glob(
                config_dictionary.paint_calibration_properties_file_name_ending
            )
        )
        valid_calibration_file_paths = []
        for calibration_file_path in calibration_file_paths:
            try:
                with open(calibration_file_path, "r") as calibration_file:
                    calibration_data = json.load(calibration_file)
                    focal_data = calibration_data.get(
                        config_dictionary.paint_focal_spot, {}
                    )
                    if (
                        config_dictionary.paint_helios in focal_data
                        and config_dictionary.paint_utis in focal_data
                    ):
                        valid_calibration_file_paths.append(calibration_file_path)
            except Exception as error:
                print(
                    f"Warning: Skipping {calibration_file_path} due to error: {error}"
                )

        if len(valid_calibration_file_paths) < minimum_calibration_files:
            continue

        valid_calibration_file_paths = valid_calibration_file_paths[
            :minimum_calibration_files
        ]

        flux_image_file_paths = []

        ending = config_dictionary.paint_calibration_properties_file_name_ending.replace("*", "").replace(".json", "")

        for calibration_file_path in valid_calibration_file_paths:
            # file_stem = calibration_file_path.stem.replace(
            #     config_dictionary.paint_calibration_properties_file_name_ending, ""
            # )
            file_stem = calibration_file_path.stem.removesuffix(ending)
            flux_image_filename = f"{file_stem}-{flux_file_suffix}.png"
            flux_image_path = calibration_directory / flux_image_filename
            if flux_image_path.exists():
                flux_image_file_paths.append(flux_image_path)

        heliostat_data_list.append(
            (heliostat_name, valid_calibration_file_paths, flux_image_file_paths)
        )
        print(
            f"Added heliostat {heliostat_name} to list. List length = {len(heliostat_data_list)}"
        )
        if len(heliostat_data_list) >= maximum_heliostats:
            break

    return sorted(heliostat_data_list, key=lambda heliostat: heliostat[0])


def save_heliostat_list(
    heliostat_data_list: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    output_file_path: str | pathlib.Path,
) -> None:
    """
    Save a list of heliostats to a JSON file, converting pathlib.Path objects to strings.

    Parameters
    ----------
    heliostat_data_list : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        The list of heliostat data with calibration and flux paths.
    output_file_path : str or pathlib.Path
        The file path to save the list to.
    """
    output_file_path = join_safe(output_file_path, "heliostat_files.json")
    output_file_path = pathlib.Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = [
        {
            "name": heliostat_name,
            "calibrations": [
                str(calibration_path) for calibration_path in calibration_paths
            ],
            "flux_images": [str(flux_path) for flux_path in flux_paths],
        }
        for heliostat_name, calibration_paths, flux_paths in heliostat_data_list
    ]

    with open(output_file_path, "w") as output_file:
        json.dump(serializable_data, output_file, indent=2)
    print(f"Saved {len(serializable_data)} heliostat entries to {output_file_path}")


if __name__ == "__main__":


    config = load_config()
    paint_directory = pathlib.Path(config["paint_repository_base_path"])


    paint_plot_base_path = pathlib.Path(config["base_path"])

    output_json_file = join_safe(paint_plot_base_path, os.path.dirname(config["results_calibration_dict_path"]))
    parser = argparse.ArgumentParser(description="Generate heliostat list with calibration info.")
    parser.add_argument(
        "--minimum-calibrations",
        type=int,
        default=25,
        help="Minimum valid calibration files per heliostat.",
    )
    parser.add_argument(
        "--maximum-heliostats",
        type=int,
        default=100,
        help="Maximum number of heliostats to include.",
    )
    parser.add_argument(
        "--excluded-heliostats",
        "-e",
        action="append",
        default=["AA39"],
        help="Heliostat names to exclude. Repeat or use comma/space separated.",
    )
    parser.add_argument(
        "--flux-file-suffix",
        type=str,
        default="flux",
        help="Flux image file suffix (e.g., 'flux' or 'flux-centered').",
    )
    args = parser.parse_args()

    minimum_calibrations = args.minimum_calibrations
    maximum_heliostats = args.maximum_heliostats

    excluded_heliostats = set()
    for group in args.excluded_heliostats:
        excluded_heliostats.update(name for name in re.split(r"[,\s;]+", group) if name)

    flux_file_suffix = args.flux_file_suffix

    heliostat_data_list = find_heliostats_with_minimum_calibrations(
        paint_directory,
        minimum_calibration_files=minimum_calibrations,
        maximum_heliostats=maximum_heliostats,
        flux_file_suffix=flux_file_suffix,
    )

    heliostat_data_list = [
        heliostat_data
        for heliostat_data in heliostat_data_list
        if heliostat_data[0] not in excluded_heliostats
    ]
    heliostat_data_list = heliostat_data_list[:maximum_heliostats]

    print(f"Selected {len(heliostat_data_list)} heliostats:")
    for heliostat_name, calibration_paths, flux_paths in heliostat_data_list:
        print(
            f"- {heliostat_name}: {len(calibration_paths)} calibrations, {len(flux_paths)} flux images ({flux_file_suffix})"
        )

    save_heliostat_list(heliostat_data_list, output_json_file)
