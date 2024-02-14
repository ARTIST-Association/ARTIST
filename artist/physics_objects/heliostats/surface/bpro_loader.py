import csv
import pathlib

import numpy as np
import struct
from typing import cast, List, Tuple
import os

from artist import ARTIST_ROOT

Tuple3d = Tuple[np.floating, np.floating, np.floating]
Vector3d = List[np.floating]


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
        The castet vector in the enu coordinate system.
    """
    return [-vec[1], vec[0], vec[2]]


def load_bpro(
    filename: str,
    concentrator_header_struct: struct.Struct,
    facet_header_struct: struct.Struct,
    ray_struct: struct.Struct,
    verbose: bool = True,
) -> Tuple[
    Vector3d,
    List[Vector3d],
    List[Vector3d],
    List[Vector3d],
    List[List[Vector3d]],
    List[List[Vector3d]],
    List[List[Vector3d]],
    float,
    float,
]:
    """
    Load a bpro file and extract information from it.

    Parameters
    ----------
    filename : str
        The file that contains the data.
    concentrator_header_struct : struct.Struct
        The concentrator header.
    facet_header_struct : struct.Struct
        The facet header.
    ray_struct : struct.Struct
        The ray struct.
    verbose : bool
        Print option.

    Returns
    -------
    Tuple[Vector3d, List[Vector3d], List[Vector3d], List[Vector3d], List[List[Vector3d]], List[List[Vector3d]], List[List[Vector3d]], float, float,
        Information about the facets and the surface
    """
    concentrator_header_struct_len = concentrator_header_struct.size
    facet_header_struct_len = facet_header_struct.size
    ray_struct_len = ray_struct.size

    binp_loc = pathlib.Path(ARTIST_ROOT)/"measurement_data"/filename
    with open(binp_loc, "rb") as file:
        byte_data = file.read(concentrator_header_struct_len)
        concentrator_header_data = concentrator_header_struct.unpack_from(byte_data)
        if verbose:
            print("READING bpro filename: " + filename)

        hel_pos = nwu_to_enu(cast(Tuple3d, concentrator_header_data[0:3]))
        width, height = concentrator_header_data[3:5]
        n_xy = concentrator_header_data[5:7]

        n_facets = n_xy[0] * n_xy[1]
        facet_positions: List[Vector3d] = []
        facet_spans_n: List[Vector3d] = []
        facet_spans_e: List[Vector3d] = []

        positions: List[List[Vector3d]] = [[] for _ in range(n_facets)]
        directions: List[List[Vector3d]] = [[] for _ in range(n_facets)]
        ideal_normal_vecs: List[List[Vector3d]] = [[] for _ in range(n_facets)]

        for f in range(n_facets):
            byte_data = file.read(facet_header_struct_len)
            facet_header_data = facet_header_struct.unpack_from(byte_data)

            # 0 for square, 1 for round 2 triangle, ...
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

            n_rays = facet_header_data[10]

            byte_data = file.read(ray_struct_len * n_rays)
            ray_datas = ray_struct.iter_unpack(byte_data)

            for ray_data in ray_datas:
                positions[f].append(cast(Tuple3d, ray_data[:3]))
                directions[f].append(cast(Tuple3d, ray_data[3:6]))
                ideal_normal_vecs[f].append(ideal_normal)

        # Stral uses two different coordinate systems, both with a West orientation. That is why we do not need an NWU
        # to ENU cast here. However, to keep our code consistent, we cast the West direction to an East direction.
        for span_e in facet_spans_e:
            span_e[0] = -span_e[0]

    return (
        hel_pos,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        positions,
        directions,
        ideal_normal_vecs,
        width,
        height,
    )