import argparse
import os
import pathlib
import struct
import warnings
from array import array

import numpy as np
import torch
import yaml

from artist.util.environment_setup import get_device


def save_binp_from_artist_data(
    heliostat: str,
    output_path: pathlib.Path,
    heliostat_position: tuple[float, float, float],
    width: float,
    height: float,
    number_of_facets: tuple[int, int],
    axis_offset: float,
    mirror_offset: float,
    facet_translation_vectors: torch.Tensor,
    canting: torch.Tensor,
    surface_points_with_facets_list: list[torch.Tensor],
    surface_normals_with_facets_list: list[torch.Tensor],
) -> None:
    """
    Generate .binp files to be used in the ``STRAL`` software comparison.

    Parameters
    ----------
    heliostat : str
        Name of the heliostat.
    output_path : pathlib.Path
        Path to where the binary file will be saved.
    heliostat_position : tuple[float, float, float]
        Heliostat positions in the east, north, up coordinate system.
    width : float
        Width of the heliostat.
    height : float
        Height of the heliostat.
    number_of_facets : tuple[int, int]
        Number of facets.
    axis_offset : float
        Axis offset.
    mirror_offset : float
        Mirror offset.
    facet_translation_vectors : torch.Tensor
        Translation vectors for the facets.
    canting : torch.Tensor
        Canting vectors for the facets.
    surface_points_with_facets_list : list[torch.Tensor]
        Surface points per facet.
    surface_normals_with_facets_list : list[torch.Tensor]
        Surface normals per facet.
    """
    concentrator_header_struct = struct.Struct("=5f2I2f")
    facet_header_struct = struct.Struct("=i9fI")
    ray_struct = struct.Struct("=7f")

    n_facets = facet_translation_vectors.shape[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        header_data = (
            float(heliostat_position[0]),
            float(heliostat_position[1]),
            float(heliostat_position[2]),
            float(width),
            float(height),
            np.uint32(number_of_facets[0]),
            np.uint32(number_of_facets[1]),
            float(axis_offset),
            float(mirror_offset),
        )
        f.write(concentrator_header_struct.pack(*header_data))

        for facet_idx in range(n_facets):
            facet_pos = (
                facet_translation_vectors[facet_idx, :3]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            facet_vec_x = canting[facet_idx, 0, :3].cpu().numpy().astype(np.float32)
            facet_vec_y = canting[facet_idx, 1, :3].cpu().numpy().astype(np.float32)

            n_rays = surface_points_with_facets_list[facet_idx].shape[0]

            f.write(
                facet_header_struct.pack(
                    int(0),
                    float(facet_pos[0]),
                    float(facet_pos[1]),
                    float(facet_pos[2]),
                    float(facet_vec_x[0]),
                    float(facet_vec_x[1]),
                    float(facet_vec_x[2]),
                    float(facet_vec_y[0]),
                    float(facet_vec_y[1]),
                    float(facet_vec_y[2]),
                    np.uint32(n_rays),
                )
            )

            positions = (
                surface_points_with_facets_list[facet_idx]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            normals = (
                surface_normals_with_facets_list[facet_idx]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            for pos, normal in zip(positions, normals):
                ray_data = (
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(
                        (width * height) / (len(positions) * n_facets)
                    ),  # power coefficient
                )
                f.write(ray_struct.pack(*ray_data))

        tracking_dll = "STJ_HELIOKON.DLL"
        tracking_method = "TrackingBasic"

        float_array = array("f", [0.0, 0.0])
        f.write(float_array)

        f.write(len(tracking_dll).to_bytes(4, byteorder="little"))
        f.write(tracking_dll.encode("ascii"))
        f.write(len(tracking_method).to_bytes(4, byteorder="little"))
        f.write(tracking_method.encode("ascii"))

    print(
        f"Wrote .binp file for heliostat {heliostat} to {os.path.abspath(output_path)}"
    )


if __name__ == "__main__":
    """
    Generate .binp files to be used in the ``STRAL`` software comparison.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_for_stral_dir : str
        Path to the directory for the generated ``STRAL`` files.
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
    data_for_stral_dir_default = config.get("data_for_stral_dir", "./data_for_stral")

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_for_stral_dir",
        type=str,
        help="Path to JSON file containing a list of heliostat names to restrict to.",
        default=data_for_stral_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    heliostats_data_path = (
        pathlib.Path(args.data_for_stral_dir)
        / "baseline"
        / "reconstructed_heliostats_data_9.pt"
    )

    heliostats_data = torch.load(heliostats_data_path, weights_only=False)

    for heliostat_index, heliostat in enumerate(heliostats_data["names"]):
        path = pathlib.Path(args.data_for_stral_dir) / "baseline" / f"{heliostat}.binp"
        save_binp_from_artist_data(
            heliostat=heliostat,
            output_path=path,
            heliostat_position=heliostats_data["positions"][heliostat_index],
            width=heliostats_data["widths"][heliostat_index],
            height=heliostats_data["heights"][heliostat_index],
            number_of_facets=heliostats_data["number_of_facets"][heliostat_index],
            axis_offset=heliostats_data["axis_offsets"][heliostat_index],
            mirror_offset=heliostats_data["mirror_offsets"][heliostat_index],
            facet_translation_vectors=heliostats_data["facet_translations"][
                heliostat_index
            ],
            canting=heliostats_data["canting_vectors"][heliostat_index],
            surface_points_with_facets_list=heliostats_data["surface_points"][
                heliostat_index
            ],
            surface_normals_with_facets_list=heliostats_data["surface_normals"][
                heliostat_index
            ],
        )
