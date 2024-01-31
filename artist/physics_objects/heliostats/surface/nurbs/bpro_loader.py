import csv
import math
from matplotlib import pyplot as plt
import numpy as np
import struct
from typing import cast, List, Tuple
import os

import torch


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
    Load a bpro (heliostat) file and extract information from it.

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
    concentratorHeader_struct_len = concentrator_header_struct.size
    facetHeader_struct_len = facet_header_struct.size
    ray_struct_len = ray_struct.size
    # powers = []
    binp_loc = os.path.join(os.path.dirname(__file__), "MeasurementData", filename)
    with open(binp_loc, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = concentrator_header_struct.unpack_from(byte_data)
        if verbose:
            print("READING bpro filename: " + filename)
        hel_pos = nwu_to_enu(cast(Tuple3d, concentratorHeader_data[0:3]))
        width, height = concentratorHeader_data[3:5]
        # offsets = concentratorHeader_data[7:9]
        n_xy = concentratorHeader_data[5:7]

        nFacets = n_xy[0] * n_xy[1]
        # nFacets =1
        facet_positions: List[Vector3d] = []
        facet_spans_n: List[Vector3d] = []
        facet_spans_e: List[Vector3d] = []

        positions: List[List[Vector3d]] = [[] for _ in range(nFacets)]
        directions: List[List[Vector3d]] = [[] for _ in range(nFacets)]
        ideal_normal_vecs: List[List[Vector3d]] = [[] for _ in range(nFacets)]

        normals = []

        for f in range(nFacets):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = facet_header_struct.unpack_from(byte_data)

            # 0 for square, 1 for round 2 triangle, ...
            # facetshape = facetHeader_data[0]
            facet_pos = cast(Tuple3d, facetHeader_data[1:4])
            facet_vec_x = np.array(
                [
                    -facetHeader_data[5],
                    facetHeader_data[4],
                    facetHeader_data[6],
                ]
            )
            facet_vec_y = np.array(
                [
                    -facetHeader_data[8],
                    facetHeader_data[7],
                    facetHeader_data[9],
                ]
            )

            facet_vec_z = np.cross(facet_vec_x, facet_vec_y)

            facet_positions.append(facet_pos)

            facet_spans_n.append(facet_vec_x.tolist())
            facet_spans_e.append(facet_vec_y.tolist())

            ideal_normal = (facet_vec_z / np.linalg.norm(facet_vec_z)).tolist()
            normals.append(ideal_normal)
            n_rays = facetHeader_data[10]

            byte_data = file.read(ray_struct_len * n_rays)
            ray_datas = ray_struct.iter_unpack(byte_data)

            for ray_data in ray_datas:
                positions[f].append(cast(Tuple3d, ray_data[:3]))
                directions[f].append(cast(Tuple3d, ray_data[3:6]))
                ideal_normal_vecs[f].append(ideal_normal)
                # powers.append(ray_data[6])

        # Stral uses two different Coord sys, both use a west orientation we dont need a nwu to enu cast here.
        # However to keep consistent in our program we cast the west direction to east direction.
        for span_e in facet_spans_e:
            span_e[0] = -span_e[0]

    # print(positions)
    # print(facet_positions)
    # print(facet_spans_e)
    # # result_list = [[-1 * element for element in row] for row in facet_spans_e]
    # # print(result_list)
    # print(facet_spans_n)
    # normals = torch.cross(torch.Tensor(facet_spans_e), torch.Tensor(facet_spans_n))
    # print(normals)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(np.array(positions)[0, :, 0], np.array(positions)[0, :, 1], np.array(positions)[0, :, 2], alpha=0.01)
    # ax.scatter(np.array(positions)[1, :, 0], np.array(positions)[1, :, 1], np.array(positions)[1, :, 2], alpha=0.01)
    # ax.scatter(np.array(positions)[2, :, 0], np.array(positions)[2, :, 1], np.array(positions)[2, :, 2], alpha=0.01)
    # ax.scatter(np.array(positions)[3, :, 0], np.array(positions)[3, :, 1], np.array(positions)[3, :, 2], alpha=0.01)

    # ax.scatter(np.array(facet_positions)[0][0], np.array(facet_positions)[0][1], np.array(facet_positions)[0][2])
    # ax.scatter(np.array(facet_positions)[1][0], np.array(facet_positions)[1][1], np.array(facet_positions)[1][2])
    # ax.scatter(np.array(facet_positions)[2][0], np.array(facet_positions)[2][1], np.array(facet_positions)[2][2])
    # ax.scatter(np.array(facet_positions)[3][0], np.array(facet_positions)[3][1], np.array(facet_positions)[3][2])

    # ax.quiver(np.array(positions)[0, :, 0], np.array(positions)[0, :, 1], np.array(positions)[0, :, 2], np.array(ideal_normal_vecs)[0, :, 0], np.array(ideal_normal_vecs)[0, :, 1], np.array(ideal_normal_vecs)[0, :, 2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(positions)[1, :, 0], np.array(positions)[1, :, 1], np.array(positions)[1, :, 2], np.array(ideal_normal_vecs)[1, :, 0], np.array(ideal_normal_vecs)[1, :, 1], np.array(ideal_normal_vecs)[1, :, 2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(positions)[2, :, 0], np.array(positions)[2, :, 1], np.array(positions)[2, :, 2], np.array(ideal_normal_vecs)[2, :, 0], np.array(ideal_normal_vecs)[2, :, 1], np.array(ideal_normal_vecs)[2, :, 2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(positions)[3, :, 0], np.array(positions)[3, :, 1], np.array(positions)[3, :, 2], np.array(ideal_normal_vecs)[3, :, 0], np.array(ideal_normal_vecs)[3, :, 1], np.array(ideal_normal_vecs)[3, :, 2], arrow_length_ratio=0, length=600)

    # ax.quiver(np.array(facet_positions)[0][0], np.array(facet_positions)[0][1], np.array(facet_positions)[0][2], normals[0][0], normals[0][1], normals[0][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[1][0], np.array(facet_positions)[1][1], np.array(facet_positions)[1][2], normals[1][0], normals[1][1], normals[1][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[2][0], np.array(facet_positions)[2][1], np.array(facet_positions)[2][2], normals[2][0], normals[2][1], normals[2][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[3][0], np.array(facet_positions)[3][1], np.array(facet_positions)[3][2], normals[3][0], normals[3][1], normals[3][2], arrow_length_ratio=0, length=600)

    # cross product
    # ax.quiver(np.array(facet_positions)[0][0], np.array(facet_positions)[0][1], np.array(facet_positions)[0][2], normals[3][0], normals[3][1], normals[3][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[1][0], np.array(facet_positions)[1][1], np.array(facet_positions)[1][2], normals[1][0], normals[1][1], normals[1][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[2][0], np.array(facet_positions)[2][1], np.array(facet_positions)[2][2], normals[2][0], normals[2][1], normals[2][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[3][0], np.array(facet_positions)[3][1], np.array(facet_positions)[3][2], normals[0][0], normals[0][1], normals[0][2], arrow_length_ratio=0, length=600)

    # ideal normals
    # ax.quiver(np.array(facet_positions)[0][0], np.array(facet_positions)[0][1], np.array(facet_positions)[0][2], normals[1][0], normals[1][1], normals[1][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[1][0], np.array(facet_positions)[1][1], np.array(facet_positions)[1][2], normals[3][0], normals[3][1], normals[3][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[2][0], np.array(facet_positions)[2][1], np.array(facet_positions)[2][2], normals[0][0], normals[0][1], normals[0][2], arrow_length_ratio=0, length=600)
    # ax.quiver(np.array(facet_positions)[3][0], np.array(facet_positions)[3][1], np.array(facet_positions)[3][2], normals[2][0], normals[2][1], normals[2][2], arrow_length_ratio=0, length=600)

    # ax.axes.set_xlim3d(left=-3, right=3)
    # ax.axes.set_ylim3d(bottom=-3, top=3)
    # ax.axes.set_zlim3d(bottom=0.00, top=200)

    # plt.show()

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


def load_csv(path: str, num_facets: int) -> List[List[Vector3d]]:
    """
    Load data from csv file.

    Parameters
    ----------
    path : str
        The path to the csv file.
    num_facets : int
        The number of facets that are to be loaded.

    Returns
    -------
    List[List[Vector3d]]
        The facets.
    """
    facets: List[List[Vector3d]] = [[] for _ in range(num_facets)]
    # mm to m conversion factor
    mm_to_m_factor = 0.001
    path = os.path.join(os.path.dirname(__file__), "MeasurementData", path)
    with open(path, "r", newline="") as csv_file:
        # Skip title
        next(csv_file)
        # Skip empty line
        next(csv_file)
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["z-integrated(mm)"] == "NaN":
                continue

            # Convert mm to m
            x = mm_to_m_factor * float(row["x-ideal(mm)"])
            y = mm_to_m_factor * float(row["y-ideal(mm)"])
            z = mm_to_m_factor * float(row["z-integrated(mm)"])
            # Facet indices in CSV start at one.
            facet_index = int(row["FacetIndex"]) - 1
            facets[facet_index].append(nwu_to_enu(cast(Tuple3d, (x, y, z))))

    return facets
