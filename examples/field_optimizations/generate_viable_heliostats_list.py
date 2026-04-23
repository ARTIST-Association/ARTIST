import argparse
import json
import pathlib
import warnings

import pandas as pd
import yaml


def find_calibration_data(
    metadata_file: pathlib.Path,
    heliostat_names: list[str] | None,
    date: str,
    maximum_number_of_measurements: int,
    kinematic_reconstruction_image_type: str,
    surface_reconstruction_image_type: str,
    excluded_heliostats: set[str] | None,
    data_dir: pathlib.Path,
) -> list[
    tuple[str, list[pathlib.Path], list[pathlib.Path], list[pathlib.Path], pathlib.Path]
]:
    """
    Select calibration file paths for heliostats based on metadata filtering.

    The function reads a metadata CSV file and selects calibration entries per heliostat.
    If `heliostat_names` is provided, only those heliostats are considered and entries
    are filtered to the exact date (YYYY-MM-DD match). If `heliostat_names` is None,
    all heliostats are considered and the function selects the `maximum_number_of_measurements`
    calibration entries closest in time to the provided date per heliostat.

    Parameters
    ----------
    metadata_file : pathlib.Path
        Path to the CSV file containing metadata of all available calibration data.
    heliostat_names : list[str] | None
        Optional list of heliostat identifiers to include. If None, all heliostats are included (default is None).
    date : str
        Reference date in 'YYYY-MM-DD' format. Used either for exact day filtering
        (when heliostat_names is provided) or as a reference for selecting the closest
        calibration entries in time (when heliostat_names is None).
    maximum_number_of_measurements : int
        Maximum number of calibration entries to select per heliostat when performing nearest-time selection.
    kinematic_reconstruction_image_type : str
        The type of calibration image to use for the kinematic reconstruction, i.e., ''flux'', or ''flux-centered''.
    surface_reconstruction_image_type : str
        The type of calibration image to use for the surface reconstruction, i.e., ''flux'', or ''flux-centered''.
    excluded_heliostats : set[str] | None
        Heliostats to exclude from processing. This filter is applied globally and overrides `heliostat_names` selection (default is None).
    data_dir : pathlib.Path
        Path to the data directory containing the calibration and heliostat data.

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
    df = pd.read_csv(metadata_file)
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    if excluded_heliostats:
        df = df[~df["HeliostatId"].isin(excluded_heliostats)]

    if heliostat_names is not None:
        df = df[df["HeliostatId"].isin(heliostat_names)]
        df = df[df["DateTime"].dt.strftime("%Y-%m-%d").str.startswith(date)]
        grouped = df.groupby("HeliostatId")["Id"].apply(list).to_dict()
    else:
        target_date = pd.to_datetime(date, utc=True)
        df["time_diff"] = (df["DateTime"] - target_date).abs()
        grouped = (
            df.sort_values("time_diff")
            .groupby("HeliostatId")
            .head(maximum_number_of_measurements)
            .groupby("HeliostatId")["Id"]
            .apply(list)
            .to_dict()
        )

    data_mapping = []
    for heliostat_id, id_list in grouped.items():
        calibration_properties = []
        kinematics_reconstruction_fluxes = []
        surface_reconstruction_fluxes = []

        valid_ids = []

        for id_ in id_list:
            calibration_properties_path = (
                data_dir
                / heliostat_id
                / "Calibration"
                / f"{id_}-calibration-properties.json"
            )
            kinematic_flux_path = (
                data_dir
                / heliostat_id
                / "Calibration"
                / f"{id_}-{kinematic_reconstruction_image_type}.png"
            )
            surface_flux_path = (
                data_dir
                / heliostat_id
                / "Calibration"
                / f"{id_}-{surface_reconstruction_image_type}.png"
            )

            if (
                calibration_properties_path.exists()
                and kinematic_flux_path.exists()
                and surface_flux_path.exists()
            ):
                valid_ids.append(id_)
                calibration_properties.append(calibration_properties_path)
                kinematics_reconstruction_fluxes.append(kinematic_flux_path)
                surface_reconstruction_fluxes.append(surface_flux_path)

        if not valid_ids:
            continue

        data_mapping.append(
            (
                heliostat_id,
                calibration_properties,
                kinematics_reconstruction_fluxes,
                surface_reconstruction_fluxes,
                data_dir
                / heliostat_id
                / "Properties"
                / f"{heliostat_id}-heliostat-properties.json",
            )
        )

    return data_mapping


if __name__ == "__main__":
    """
    Generate list of viable heliostats for the field optimizations.

    This script identifies a list of viable heliostats, i.e., containing a minimum number of valid measurements, for
    the optimization process. It will create one list for the baseline case, including the 96 heliostats specified in
    the `config.yaml`, and one list for the full-field case, including all heliostats with the minimum amount of calibration
    files available.

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
    excluded_heliostats_for_reconstruction : list[str]
        Heliostat names to exclude from the reconstruction process.
    heliostat_list_baseline : list[str]
        List of all heliostat names included in the baseline measurement.
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
    metadata_file_default = config.get(
        "metadata_file_name", "calibration_metadata_all_heliostats.csv"
    )
    data_dir_default = config.get("data_dir", "./paint_data")
    results_dir_default = config.get(
        "results_dir", "./examples/field_optimizations/results"
    )
    maximum_number_of_measurements_default = config.get(
        "maximum_number_of_measurements", 4
    )
    date_of_measurement_default = config.get("date_of_measurement", "2021-09-16")
    kinematic_reconstruction_image_type_default = config.get(
        "kinematic_reconstruction_image_type", "flux"
    )
    surface_reconstruction_image_type_default = config.get(
        "surface_reconstruction_image_type", "flux-centered"
    )
    excluded_heliostats_default = config.get(
        "excluded_heliostats_for_reconstruction", ["BE20", "AP14", "AG21"]
    )
    heliostat_list_baseline_default = config.get("heliostat_list_baseline", None)

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        help="Name of metadata file.",
        default=metadata_file_default,
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
        "--maximum_number_of_measurements",
        type=int,
        help="Maximum number of calibration measurements per heliostat.",
        default=maximum_number_of_measurements_default,
    )
    parser.add_argument(
        "--date_of_measurement",
        type=str,
        help="Date of the calibration measurement.",
        default=date_of_measurement_default,
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
    parser.add_argument(
        "--excluded_heliostats_for_reconstruction",
        type=str,
        help="Heliostat names to exclude.",
        nargs="+",
        default=excluded_heliostats_default,
    )
    parser.add_argument(
        "--heliostat_list_baseline",
        type=list[str],
        help="List of all heliostat names included in the baseline measurement.",
        default=heliostat_list_baseline_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)
    metadata_file = pathlib.Path(
        "./examples/field_optimizations/metadata/", args.metadata_file_name
    )
    data_dir = pathlib.Path(args.data_dir)
    excluded_heliostats: set[str] = set(args.excluded_heliostats_for_reconstruction)

    for case in ["baseline", "full_field"]:
        if case == "baseline":
            heliostat_list = args.heliostat_list_baseline
        else:
            heliostat_list = None

        heliostat_data_list = find_calibration_data(
            metadata_file=metadata_file,
            heliostat_names=heliostat_list,
            date=args.date_of_measurement,
            maximum_number_of_measurements=args.maximum_number_of_measurements,
            kinematic_reconstruction_image_type=args.kinematic_reconstruction_image_type,
            surface_reconstruction_image_type=args.surface_reconstruction_image_type,
            excluded_heliostats=excluded_heliostats,
            data_dir=data_dir,
        )

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
                    str(flux_path)
                    for flux_path in surface_reconstruction_flux_image_path
                ],
                "properties": str(properties_path),
            }
            for heliostat_name, calibration_paths, kinematic_reconstruction_flux_paths, surface_reconstruction_flux_image_path, properties_path in heliostat_data_list
        ]

        results_path = pathlib.Path(args.results_dir) / case / "viable_heliostats.json"

        if not results_path.parent.is_dir():
            results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as output_file:
            json.dump(serializable_data, output_file, indent=2)

        print(f"Saved {len(serializable_data)} heliostat entries to {results_path}")
