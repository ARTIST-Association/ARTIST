import json
import os
import pathlib
import torch

from artist.util.environment_setup import get_device


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

def load_config():
    """Load local example configuration from config.local.json (same pattern as 01/02)."""
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, "config.local.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config.local.json found. Copy config.example.json to config.local.json and customize it."
    ) 

def join_safe(base: pathlib.Path, maybe_rel: str | pathlib.Path) -> pathlib.Path:
    """Join base with possibly relative path, stripping leading separators."""
    s = str(maybe_rel)
    return base / s.lstrip("/\\")

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

@staticmethod
def perform_inverse_canting_and_translation(
    canted_points: torch.Tensor,
    translation: torch.Tensor,
    canting: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Invert the canting rotation and translation on a batch of facets.

    Parameters
    ----------
    canted_points : torch.Tensor
        Homogeneous points after the forward transform, shape (number_of_facets, number_of_points, 4).
    translation : torch.Tensor
        Batch of facet translations, shape (number_of_facets, 4).
    canting : torch.Tensor
        Batch of canting vectors (east, north), shape (number_of_facets, 2, 4).
    device : torch.device | None
        Computation device.

    Returns
    -------
    torch.Tensor
        Original 3D points, shape (number_of_facets, number_of_points, 3).
    """
    device = get_device(device=device)
    number_of_facets, number_of_points, _ = canted_points.shape

    # Build forward transform per facet (use only ENU 3D for rotation).
    forward_transform = torch.zeros((number_of_facets, 4, 4), device=device)

    east_unit_vector = torch.nn.functional.normalize(
        canting[:, 0, :3], dim=1
    )  # (F, 3).
    north_unit_vector = torch.nn.functional.normalize(
        canting[:, 1, :3], dim=1
    )  # (F, 3).
    up_unit_vector = torch.nn.functional.normalize(
        torch.linalg.cross(east_unit_vector, north_unit_vector, dim=1), dim=1
    )  # (F, 3).

    forward_transform[:, :3, 0] = east_unit_vector
    forward_transform[:, :3, 1] = north_unit_vector
    forward_transform[:, :3, 2] = up_unit_vector
    # Translation column; ensure bottom element is 1.
    forward_transform[:, :3, 3] = translation[:, :3]
    forward_transform[:, 3, 3] = 1.0

    # Extract rotation and translation.
    rotation_matrix = forward_transform[:, :3, :3]  # (F, 3, 3).
    translation_vector = forward_transform[:, :3, 3]  # (F, 3).

    # Compute inverse transform.
    rotation_matrix_inverse = rotation_matrix.transpose(1, 2)  # (F, 3, 3).
    translation_inverse = -torch.bmm(
        rotation_matrix_inverse, translation_vector.unsqueeze(-1)
    ).squeeze(-1)  # (F, 3).

    inverse_transform = torch.zeros((number_of_facets, 4, 4), device=device)
    inverse_transform[:, :3, :3] = rotation_matrix_inverse
    inverse_transform[:, :3, 3] = translation_inverse
    inverse_transform[:, 3, 3] = 1.0

    # Apply inverse transform.
    restored_points = torch.bmm(canted_points, inverse_transform.transpose(1, 2))
    return restored_points[..., :3]



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
