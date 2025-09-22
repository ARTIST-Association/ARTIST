import json
import os
import pathlib
import warnings

import torch

from artist.util import config_dictionary, utils
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
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

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


def extract_canting_and_translation_from_properties(
    heliostat_list: list[
        tuple[str, pathlib.Path] | tuple[str, pathlib.Path, pathlib.Path]
    ],
    convert_to_4d: bool = False,
    device: torch.device | None = None,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Extract facet translation and canting vectors per heliostat from ``PAINT`` properties files.

    Parameters
    ----------
    heliostat_list : list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
        A list where each entry is either a tuple containing the heliostat name and the path to the heliostat properties
        data, or the heliostat name, path to the properties file, and path to the deflectometry data.
    convert_to_4d : bool
        Indicating whether tensors should be converted to 4D format (default is `False`).
    device : torch.device | None
        The device on which to create tensors (default is None).

    Returns
    -------
    list[tuple[str, torch.Tensor, torch.Tensor]]
        A list containing a tupe for each heliostat including the heliostat name,the facet translations tensor of shape
        [num_facets, 3] and the facet canting tensor of shape [num_facets, 2, 3].
    """
    device = get_device(device=device)
    facet_transforms_per_heliostat: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    for entry in heliostat_list:
        try:
            heliostat_name = str(entry[0])
            properties_path = pathlib.Path(entry[1])

            with open(properties_path, "r") as f:
                heliostat_dict = json.load(f)

            num_facets = heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_number_of_facets
            ]

            # Allocate tensors (ENU 3D by default)
            facet_translations_enu3 = torch.empty((num_facets, 3), device=device)
            facet_canting_vectors_enu3 = torch.empty((num_facets, 2, 3), device=device)

            # Fill from JSON
            for facet_idx in range(num_facets):
                facet_entry = heliostat_dict[config_dictionary.paint_facet_properties][
                    config_dictionary.paint_facets
                ][facet_idx]
                facet_translations_enu3[facet_idx, :] = torch.tensor(
                    facet_entry[config_dictionary.paint_translation_vector],
                    device=device,
                )
                facet_canting_vectors_enu3[facet_idx, 0] = torch.tensor(
                    facet_entry[config_dictionary.paint_canting_e], device=device
                )
                facet_canting_vectors_enu3[facet_idx, 1] = torch.tensor(
                    facet_entry[config_dictionary.paint_canting_n], device=device
                )

            # Optional conversion to homogeneous 4D
            if convert_to_4d:
                facet_translations = utils.convert_3d_directions_to_4d_format(
                    facet_translations_enu3, device=device
                )
                facet_canting_vectors = utils.convert_3d_directions_to_4d_format(
                    facet_canting_vectors_enu3, device=device
                )
            else:
                facet_translations = facet_translations_enu3
                facet_canting_vectors = facet_canting_vectors_enu3

            facet_transforms_per_heliostat.append(
                (heliostat_name, facet_translations, facet_canting_vectors)
            )

        except Exception as ex:
            warnings.warn(
                f"Failed to extract canting/translation for '{entry[0]}' "
                f"from properties '{entry[1]}': {ex}"
            )
            continue

    return facet_transforms_per_heliostat


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
