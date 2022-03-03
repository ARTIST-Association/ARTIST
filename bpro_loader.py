import struct
from typing import List, Tuple

import numpy as np

Vector3d = List[np.float64]


def load_bpro(
        filename: str,
        concentratorHeader_struct: struct.Struct,
        facetHeader_struct: struct.Struct,
        ray_struct: struct.Struct,
) -> Tuple[
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
    with open(filename, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = concentratorHeader_struct.unpack_from(
            byte_data)
        print("READING bpro filename: " + filename)

        # hel_pos = concentratorHeader_data[0:3]
        width, height = concentratorHeader_data[3:5]
        # offsets = concentratorHeader_data[7:9]
        n_xy = concentratorHeader_data[5:7]

        nFacets = n_xy[0] * n_xy[1]
        # nFacets =1
        positions: List[List[Vector3d]] = [[] for _ in range(nFacets)]
        directions: List[List[Vector3d]] = [[] for _ in range(nFacets)]
        ideal_normal_vecs: List[List[Vector3d]] = [
            [] for _ in range(nFacets)
        ]

        for f in range(nFacets):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = facetHeader_struct.unpack_from(byte_data)

            # 0 for square, 1 for round 2 triangle, ...
            # facetshape = facetHeader_data[0]
            # facet_pos = facetHeader_data[1:4]
            facet_vec_x = np.array(facetHeader_data[4:7])
            facet_vec_y = np.array(facetHeader_data[7:10])
            facet_vec_z = np.cross(facet_vec_x, facet_vec_y).tolist()
            # print("X", facet_vec_x)
            # print("Y", facet_vec_y)
            n_rays = facetHeader_data[10]

            byte_data = file.read(ray_struct_len * n_rays)
            ray_datas = ray_struct.iter_unpack(byte_data)

            for ray_data in ray_datas:
                positions[f].append([ray_data[0], ray_data[1], ray_data[2]])
                directions[f].append([ray_data[3], ray_data[4], ray_data[5]])
                ideal_normal_vecs[f].append(facet_vec_z)
                # powers.append(ray_data[6])

    return positions, directions, ideal_normal_vecs, width, height
