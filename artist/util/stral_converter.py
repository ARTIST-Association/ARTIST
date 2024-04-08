import logging
import struct
import sys
from typing import List, Tuple, cast

import colorlog
import h5py
import numpy as np
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary

Tuple3d = Tuple[np.floating, np.floating, np.floating]
Vector3d = List[np.floating]


def get_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger for the conversion process.

    Returns
    -------
    logging.Logger
        The logger for the conversion process
    """
    log = logging.getLogger("STRAL-to-h5-converter")  # Get logger instance.
    log_formatter = colorlog.ColoredFormatter(
        fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
        "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(log_formatter)
    log.addHandler(handler)
    log.setLevel(log_level)
    return log


def convert_point_to_4d_format(point: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3d point to a 4d point.

    Parameters
    ----------
    point : torch.Tensor
        Point in 3D format.

    Returns
    -------
    torch.Tensor
        The point in a 4D format, with an appended one.

    """
    if len(point.size()) == 1:
        return torch.cat((point, torch.ones(1)), dim=0)
    else:
        return torch.cat((point, torch.ones(point.size(0), 1)), dim=1)


def convert_direction_to_4d_format(direction: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3d direction vector to a 4d direction vector.

    Parameters
    ----------
    direction : torch.Tensor
        Direction vector in 3D format.

    Returns
    -------
    torch.Tensor
        The direction vector in a 4D format, with an appended zero.

    """
    if len(direction.size()) == 1:
        return torch.cat((direction, torch.zeros(1)), dim=0)
    else:
        return torch.cat((direction, torch.zeros(direction.size(0), 1)), dim=1)


def nwu_to_enu(vec: Tuple3d) -> Vector3d:
    """
    Cast the coordinate system from nwu to enu.

    Parameters
    ----------
    vec : Tuple3d
        The vector that is to be casted.

    Returns
    -------
    Vector3d
        The casted vector in the enu coordinate system.
    """
    return [-vec[1], vec[0], vec[2]]


def convert_stral_to_h5(
    stral_file_path: str,
    hdf5_file_path: str,
    concentrator_header_name: str,
    facet_header_name: str,
    ray_struct_name: str,
    step_size: int,
    log: logging.Logger,
) -> None:
    """
    Extract information from a STRAL file saved as .binp and save this information as a HDF5 file.

    Parameters
    ----------
    stral_file_path : str
        The file path to the STRAL data file that will be converted
    hdf5_file_path : str
        The file path for the HDF5 file that will be saved
    concentrator_header_name : str
        The name for the concentrator header in the STRAL file
    facet_header_name : str
        The name for the facet header in the STRAL file
    ray_struct_name : str
        The name of the ray structure in the STRAL file
    step_size : int
        The size of the step used to reduce the number of considered points for compute efficiency
    log : logging.Logger
        The logger
    """
    log.info("Beginning STRAL to HDF5 conversion!")

    # Create structures for reading STRAL file correctly
    concentrator_header_struct = struct.Struct(concentrator_header_name)
    facet_header_struct = struct.Struct(facet_header_name)
    ray_struct = struct.Struct(ray_struct_name)

    with open(f"{stral_file_path}.binp", "rb") as file:
        byte_data = file.read(concentrator_header_struct.size)
        concentrator_header_data = concentrator_header_struct.unpack_from(byte_data)
        log.info(f"Reading STRAL file located at: {stral_file_path}")

        # Load surface position
        surface_position = nwu_to_enu(cast(Tuple3d, concentrator_header_data[0:3]))

        # Load width and height
        width, height = concentrator_header_data[3:5]

        # Calculate number of facets
        n_xy = concentrator_header_data[5:7]
        number_of_facets = n_xy[0] * n_xy[1]

        # Create empty lists for storing data
        facet_positions: List[Vector3d] = []
        facet_spans_n: List[Vector3d] = []
        facet_spans_e: List[Vector3d] = []
        surface_points: List[List[Vector3d]] = [[] for _ in range(number_of_facets)]
        surface_normals: List[List[Vector3d]] = [[] for _ in range(number_of_facets)]
        surface_ideal_vectors: List[List[Vector3d]] = [
            [] for _ in range(number_of_facets)
        ]

        normals = []

        for f in range(number_of_facets):
            byte_data = file.read(facet_header_struct.size)
            facet_header_data = facet_header_struct.unpack_from(byte_data)

            facet_pos = cast(Tuple3d, facet_header_data[1:4])
            facet_vec_x = np.array(
                [
                    -facet_header_data[5],
                    facet_header_data[4],
                    facet_header_data[6],
                ]
            )
            facet_vec_y = np.array(
                [
                    -facet_header_data[8],
                    facet_header_data[7],
                    facet_header_data[9],
                ]
            )

            facet_vec_z = np.cross(facet_vec_x, facet_vec_y)

            facet_positions.append(facet_pos)

            facet_spans_n.append(facet_vec_x.tolist())
            facet_spans_e.append(facet_vec_y.tolist())

            ideal_normal = (facet_vec_z / np.linalg.norm(facet_vec_z)).tolist()
            normals.append(ideal_normal)
            n_rays = facet_header_data[10]

            byte_data = file.read(ray_struct.size * n_rays)
            ray_datas = ray_struct.iter_unpack(byte_data)

            for ray_data in ray_datas:
                surface_points[f].append(cast(Tuple3d, ray_data[:3]))
                surface_normals[f].append(cast(Tuple3d, ray_data[3:6]))
                surface_ideal_vectors[f].append(ideal_normal)

    log.info("Loading STRAL data complete")

    # Stral uses two different coordinate systems, both use a west orientation and therefore, we dont need a nwu to enu
    # cast here. However, to maintain consistency we cast the west direction to east direction.
    for span_e in facet_spans_e:
        span_e[0] = -span_e[0]

    # Reshape and select only selected number of points to reduce compute
    surface_points = torch.tensor(surface_points).view(-1, 3)[::step_size]
    surface_normals = torch.tensor(surface_normals).view(-1, 3)[::step_size]
    surface_ideal_vectors = torch.tensor(surface_ideal_vectors).view(-1, 3)[::step_size]

    # Convert to torch tensor
    surface_position = torch.tensor(surface_position)
    facet_positions = torch.tensor(facet_positions)
    facet_spans_n = torch.tensor(facet_spans_n)
    facet_spans_e = torch.tensor(facet_spans_e)

    # Convert to 4D Format
    surface_position = convert_point_to_4d_format(surface_position)
    facet_positions = convert_point_to_4d_format(facet_positions)
    facet_spans_n = convert_direction_to_4d_format(facet_spans_n)
    facet_spans_e = convert_direction_to_4d_format(facet_spans_e)
    surface_points = convert_point_to_4d_format(surface_points)
    surface_normals = convert_direction_to_4d_format(surface_normals)
    surface_ideal_vectors = convert_direction_to_4d_format(surface_ideal_vectors)

    log.info(f"Creating HDF5 file in location: {hdf5_file_path}")

    with h5py.File(f"{hdf_file_path}.h5", "w") as f:
        f[config_dictionary.load_surface_position_key] = surface_position
        f[config_dictionary.load_facet_positions_key] = facet_positions
        f[config_dictionary.load_facet_spans_n_key] = facet_spans_n
        f[config_dictionary.load_facet_spans_e_key] = facet_spans_e
        f[config_dictionary.load_points_key] = surface_points
        f[config_dictionary.load_normals_key] = surface_normals
        f[config_dictionary.load_surface_ideal_vectors_key] = surface_ideal_vectors


# Variables that must be set for the converter to work
stral_file_path = f"{ARTIST_ROOT}/STRAL_data/stral_test_data"
hdf_file_path = (
    f"{ARTIST_ROOT}/{config_dictionary.measurement_location}/stral_conversion_test"
)
concentrator_header_name = "=5f2I2f"
facet_header_name = "=i9fI"
ray_struct_name = "=7f"
step_size = 2

if __name__ == "__main__":
    log = get_logger()

    convert_stral_to_h5(
        stral_file_path=stral_file_path,
        hdf5_file_path=hdf_file_path,
        concentrator_header_name=concentrator_header_name,
        facet_header_name=facet_header_name,
        ray_struct_name=ray_struct_name,
        step_size=step_size,
        log=log,
    )
