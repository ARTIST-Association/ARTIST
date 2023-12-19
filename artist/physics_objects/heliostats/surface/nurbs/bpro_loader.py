import csv
import numpy as np
import struct
from typing import cast, List, Tuple
import os


Tuple3d = Tuple[np.floating, np.floating, np.floating]
Vector3d = List[np.floating]


def nwu_to_enu(vec: Tuple3d) -> Vector3d:
    return [-vec[1], vec[0], vec[2]]


def load_bpro(
    filename: str,
    concentratorHeader_struct: struct.Struct,
    facetHeader_struct: struct.Struct,
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
    concentratorHeader_struct_len = concentratorHeader_struct.size
    facetHeader_struct_len = facetHeader_struct.size
    ray_struct_len = ray_struct.size

    # powers = []
    binp_loc = os.path.join(os.path.dirname(__file__), "MeasurementData", filename)
    with open(binp_loc, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = concentratorHeader_struct.unpack_from(byte_data)
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

        for f in range(nFacets):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = facetHeader_struct.unpack_from(byte_data)

            # 0 for square, 1 for round 2 triangle, ...
            # facetshape = facetHeader_data[0]
            facet_pos = cast(Tuple3d, facetHeader_data[1:4])
            # print(facetHeader_data[1:4])
            # print(facet_pos)
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
