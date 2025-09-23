import json
import pathlib

import matplotlib as mpl


def load_heliostat_data(
    paint_repo: str | pathlib.Path, input_path: str | pathlib.Path
) -> tuple[
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    list[tuple[str, pathlib.Path]],
]:
    """Load heliostat calibration/flux-image mapping and derive properties file paths.

    Parameters
    ----------
    paint_repo : str | pathlib.Path
        Base path to the paint repository containing heliostat folders.
    input_path : str | pathlib.Path
        Path to a JSON file with entries:
        [
          {"name": "<heliostat_name>", "calibrations": ["..."], "flux_images": ["..."]},
          ...
        ]

    Returns
    -------
    tuple
        A tuple of:
        - heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
          For each heliostat, (name, list_of_calibration_property_paths, list_of_flux_image_paths).
        - heliostat_properties_list : list[tuple[str, pathlib.Path]]
          For each heliostat, (name, properties_json_path). Only existing files are included.

    Notes
    -----
    Paths in the JSON are resolved to absolute `pathlib.Path` instances.
    """
    input_path = pathlib.Path(input_path).resolve()
    paint_repo = pathlib.Path(paint_repo).resolve()

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    heliostat_data_mapping: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ] = []
    heliostat_properties_list: list[tuple[str, pathlib.Path]] = []

    for item in raw_data:
        name = item["name"]
        calibrations = [pathlib.Path(p) for p in item["calibrations"]]
        flux_images = [pathlib.Path(p) for p in item["flux_images"]]

        heliostat_data_mapping.append((name, calibrations, flux_images))

        properties_path = (
            paint_repo / name / "Properties" / f"{name}-heliostat-properties.json"
        )
        if properties_path.exists():
            heliostat_properties_list.append((name, properties_path))
        else:
            print(f"Warning: Missing properties file for {name} at {properties_path}")

    return heliostat_data_mapping, heliostat_properties_list


def filter_valid_heliostat_data(
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """Filter flux images so each has a matching calibration stem per heliostat.

    Parameters
    ----------
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        For each heliostat, (name, calibration_property_paths, flux_image_paths).

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Filtered mapping with only flux images that match available calibration stems.
    """
    valid = []
    for heliostat_name, valid_calibrations, flux_paths in heliostat_data_mapping:
        valid_stems = {
            p.stem.replace("-calibration-properties", "") for p in valid_calibrations
        }
        valid_flux_paths = [
            f for f in flux_paths if f.stem.replace("-flux", "") in valid_stems
        ]
        valid.append((heliostat_name, valid_calibrations, valid_flux_paths))

    print("\nFiltered Heliostat Data Mapping:")
    for name, calibs, fluxes in valid:
        print(
            f"- {name}: {len(calibs)} valid calibrations, {len(fluxes)} matching flux images"
        )
    return valid


def set_plot_style() -> None:
    """Set global plot style for all plots."""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10

    # Now, enable LaTeX and tell it to use a sans-serif font.
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = (
        "\\usepackage{helvet}\\usepackage{sansmath}\\sansmath"
    )


# Plot Settings.
helmholtz_colors = {
    "hgfblue": "#005AA0",
    "hgfdarkblue": "#0A2D6E",
    "hgfgreen": "#8CB423",
    "hgfgray": "#5A696E",
    "hgfaerospace": "#50C8AA",
    "hgfearthandenvironment": "#326469",
    "hgfenergy": "#FFD228",
    "hgfhealth": "#D23264",
    "hgfkeytechnologies": "#A0235A",
    "hgfmatter": "#F0781E",
}
