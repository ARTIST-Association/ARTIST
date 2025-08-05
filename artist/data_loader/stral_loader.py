import logging
import pathlib
import struct

import torch

from artist.util import utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the stral data loader."""


def extract_stral_deflectometry_data(
    stral_file_path: pathlib.Path,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract deflectometry data from a ```STRAL`` file.

    Parameters
    ----------
    stral_file_path : pathlib.Path
        The file path to the ``STRAL`` data that will be converted.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The facet translation vectors.
        Tensor of shape [number_of_facets, 4].
    torch.Tensor
        The facet canting vectors.
        Tensor of shape [number_of_facets, 2, 4].
    list[torch.Tensor]
        The surface points per facet.
    list[torch.Tensor]
        The surface normals per facet.
    """
    device = get_device(device=device)

    log.info("Beginning extraction of data from ```STRAL``` file.")

    # Create structures for reading ``STRAL`` file.
    surface_header_struct = struct.Struct("=5f2I2f")
    facet_header_struct = struct.Struct("=i9fI")
    points_on_facet_struct = struct.Struct("=7f")
    log.info(f"Reading STRAL file located at: {stral_file_path}")
    with open(f"{stral_file_path}", "rb") as file:
        surface_header_data = surface_header_struct.unpack_from(
            file.read(surface_header_struct.size)
        )

        # Calculate the number of facets.
        n_xy = surface_header_data[5:7]
        number_of_facets = n_xy[0] * n_xy[1]

        # Create empty tensors for storing data.
        facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
        canting = torch.empty(number_of_facets, 2, 3, device=device)
        surface_points_with_facets_list = []
        surface_normals_with_facets_list = []
        for facet in range(number_of_facets):
            facet_header_data = facet_header_struct.unpack_from(
                file.read(facet_header_struct.size)
            )
            facet_translation_vectors[facet] = torch.tensor(
                facet_header_data[1:4], dtype=torch.float, device=device
            )
            canting[facet, 0] = torch.tensor(
                facet_header_data[4:7], dtype=torch.float, device=device
            )
            canting[facet, 1] = torch.tensor(
                facet_header_data[7:10], dtype=torch.float, device=device
            )
            number_of_points = facet_header_data[10]
            single_facet_surface_points = torch.empty(
                number_of_points, 3, device=device
            )
            single_facet_surface_normals = torch.empty(
                number_of_points, 3, device=device
            )

            points_data = points_on_facet_struct.iter_unpack(
                file.read(points_on_facet_struct.size * number_of_points)
            )
            for i, point_data in enumerate(points_data):
                single_facet_surface_points[i, :] = torch.tensor(
                    point_data[:3], dtype=torch.float, device=device
                )
                single_facet_surface_normals[i, :] = torch.tensor(
                    point_data[3:6], dtype=torch.float, device=device
                )
            surface_points_with_facets_list.append(single_facet_surface_points)
            surface_normals_with_facets_list.append(single_facet_surface_normals)

        # Convert to 4D format.
        facet_translation_vectors = utils.convert_3d_directions_to_4d_format(
            facet_translation_vectors, device=device
        )
        canting = utils.convert_3d_directions_to_4d_format(canting, device=device)


    log.info("Loading ``STRAL`` data complete.")

    return (
        facet_translation_vectors,
        canting,
        surface_points_with_facets_list,
        surface_normals_with_facets_list,
    )
