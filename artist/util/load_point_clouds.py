import json
import math
import logging
import pathlib
import struct
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt #TODO Remove Later

import torch
import torch.nn.functional as F

from artist.util.loading_files_utils import load_heliostat_properties_file
from artist.util import config_dictionary, loading_files_utils, utils
from artist.util.configuration_classes import FacetConfig
from artist.util.surface_converter import AnalyticalConfig
    

log = logging.getLogger(__name__)
def _try_load_deflectometry(
        deflectometry_file_path: Optional[str],
        config_dictionary: dict,
        number_of_facets: int,
        device: torch.device
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if deflectometry_file_path and pathlib.Path(deflectometry_file_path).is_file():
            log.info(f"Reading PAINT deflectometry file located at: {deflectometry_file_path}.")
            surface_points_with_facets_list, surface_normals_with_facets_list = loading_files_utils.load_surface_points_and_normals_from_h5(
                deflectometry_file_path, config_dictionary, number_of_facets, device
            )
            log.info("Loading PAINT with deflectometry data complete.")
            return surface_points_with_facets_list, surface_normals_with_facets_list
        else:
            log.warning(f"Deflectometry file not found or invalid at: {deflectometry_file_path}.")
            return None
def _try_load_ideal_tensor(
    ideal_file: torch.Tensor,
    number_of_facets: int,
    device: torch.device
) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    if ideal_file.ndim != 4 or ideal_file.shape[0] != 2 or ideal_file.shape[3] != 3:
        raise ValueError(
            f"Ideal tensor must have shape (2, num_facets, num_points, 3), "
            f"but got {ideal_file.shape}."
        )
    if ideal_file.shape[1] != number_of_facets:
        raise ValueError(
            f"Ideal tensor facet count mismatch: expected {number_of_facets}, "
            f"but got {ideal_file.shape[1]}."
        )

    log.info("Loading PAINT with ideal tensor surface points and normals.")
    surface_points_with_facets_list = [
        ideal_file[0, i, :, :].to(device) for i in range(number_of_facets)
    ]
    surface_normals_with_facets_list = [
        ideal_file[1, i, :, :].to(device) for i in range(number_of_facets)
    ]
    return surface_points_with_facets_list, surface_normals_with_facets_list

def _try_load_ideal_file(
    ideal_file: str,
    config_dictionary: dict,
    number_of_facets: int,
    device: torch.device
) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    if pathlib.Path(ideal_file).is_file():
        log.warning(
            "You have loaded an ideal point cloud from a file. Currently, there is no algorithm capable of generating such files. "
            "This might indicate an error in your input or processing pipeline."
        )
        log.info(f"Reading ideal point cloud file located at: {ideal_file}.")
        surface_points_with_facets_list, surface_normals_with_facets_list = loading_files_utils.load_surface_points_and_normals_from_h5(
            ideal_file, config_dictionary, number_of_facets, device
        )
        log.info("Loading PAINT with deflectometry data complete.")
        return surface_points_with_facets_list, surface_normals_with_facets_list
    return None

def generate_point_cloud_from_stral_data(stral_file_path: pathlib.Path, device: Union[torch.device, str] = "cuda"):        
        log.info("Beginning extraction of data from ```STRAL``` file.")
        device = torch.device(device)

        # Create structures for reading ``STRAL`` file.
        surface_header_struct = struct.Struct("=5f2I2f")
        facet_header_struct = struct.Struct("=i9fI")
        points_on_facet_struct = struct.Struct("=7f")
        log.info(f"Reading STRAL file located at: {stral_file_path}")
        with open(f"{stral_file_path}", "rb") as file:
            surface_header_data = surface_header_struct.unpack_from(
                file.read(surface_header_struct.size)
            )
            # Load width and heigt of the whole heliostat
            width, height = surface_header_data[3:5]
            # Calculate the number of facets.
            n_xy = surface_header_data[5:7]
            number_of_facets = n_xy[0] * n_xy[1]

            # Create empty tensors for storing data.
            facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
            canting_e = torch.empty(number_of_facets, 3, device=device)
            canting_n = torch.empty(number_of_facets, 3, device=device)
            surface_points_with_facets_list = []
            surface_normals_with_facets_list = []
            for f in range(number_of_facets):
                facet_header_data = facet_header_struct.unpack_from(
                    file.read(facet_header_struct.size)
                )
                facet_translation_vectors[f] = torch.tensor(
                    facet_header_data[1:4], dtype=torch.float, device=device
                )
                canting_e[f] = torch.tensor(
                    facet_header_data[4:7], dtype=torch.float, device=device
                )
                canting_n[f] = torch.tensor(
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

        log.info("Loading ``STRAL`` data complete.")

def create_point_cloud_with_fixed_aspect_ratio(total_heliostat_height, total_heliostat_width, 
                                               desired_points_per_facet, number_facets, device):
    """
    Creates a point cloud for heliostat facets with a fixed aspect ratio.
    Returns points of shape (number_facets, num_points, 3).
    """
    device = torch.device(device)
    
    # Define dimensions for a single facet (half the heliostat's size)
    single_facet_height = total_heliostat_height / 2
    single_facet_width  = total_heliostat_width / 2

    # --- Step 1: Determine spacing along the East direction ---
    num_points_east = int(round(math.sqrt(desired_points_per_facet)))
    num_points_east = max(num_points_east, 1)
    point_spacing = single_facet_width / (num_points_east - 1) if num_points_east > 1 else single_facet_width

    # --- Step 2: Determine the number of points along the North direction ---
    num_points_north = int(single_facet_height / point_spacing) + 1
    num_points_north = max(num_points_north, 1)
    grid_north_span = (num_points_north - 1) * point_spacing

    east_coords = torch.linspace(-single_facet_width / 2, single_facet_width / 2, 
                                 steps=num_points_east, device=device)
    north_coords = torch.linspace(-grid_north_span / 2, grid_north_span / 2, 
                                  steps=num_points_north, device=device)

    # --- Step 3: Create the grid for one facet ---
    mesh_east, mesh_north = torch.meshgrid(east_coords, north_coords, indexing='ij')
    mesh_east = mesh_east.reshape(-1)
    mesh_north = mesh_north.reshape(-1)
    mesh_up = torch.zeros_like(mesh_east, device=device)

    facet_point_cloud = torch.stack([mesh_east, mesh_north, mesh_up], dim=-1)  # (num_points, 3)

    # --- (Optional) Report the difference in point count ---
    actual_point_count = num_points_east * num_points_north
    point_difference = actual_point_count - desired_points_per_facet
    if point_difference > 0:
        log.info(f"Using {actual_point_count} points, which is {point_difference} more than requested {desired_points_per_facet}.")
    elif point_difference < 0:
        log.info(f"Using {actual_point_count} points, which is {-point_difference} less than requested {desired_points_per_facet}.")
    else:
        log.info(f"Using exactly {actual_point_count} points as requested.")

    # --- Step 4: Repeat for all facets ---
    complete_point_cloud = facet_point_cloud.unsqueeze(0).repeat(number_facets, 1, 1)
    return complete_point_cloud

def generate_ideal_juelich_heliostat_pointcloud_from_paint_heliostat_properties(
        heliostat_file_path: str,
        number_of_surface_points: int,
        device: Union[torch.device, str] = "cuda"
        ) -> List[torch.Tensor]:
        """
        Generate an ideal Jülich heliostat surface as a collection of points on 4 facets.
        """
        (
            number_of_facets,
            heliostat_height,
            heliostat_width,
            facet_translation_vectors,
            cantings_e,
            cantings_n,
        ) = load_heliostat_properties_file(heliostat_file_path= heliostat_file_path, device = device)
        ###TODO CANTING ADAPTION REMOVE LATER####
        cantings_n = torch.cat((-cantings_n[:, :2], cantings_n[:, 2:]), dim=1)
        cantings_e = torch.cat((-cantings_e[:, :2], cantings_e[:, 2:]), dim=1)

        # Compute half-dimensions from the spatial components.
        half_facet_heights = torch.norm(cantings_n, dim=1)  # shape: (4,)
        half_facet_widths  = torch.norm(cantings_e, dim=1)   # shape: (4,)
        
        # Assuming all facets are identical, pick the dimensions from the first facet.
        facet_height = 4 * half_facet_heights[0]
        facet_width  = 4 * half_facet_widths[0]
        
        number_of_facets = 4
        # Now the point cloud has shape (4, N, 3)
        surface_pointcloud = create_point_cloud_with_fixed_aspect_ratio(
            facet_height, facet_width, number_of_surface_points, number_of_facets, device)

        # Normalize the spatial part of the canting vectors (first three components)
        canting_e_spatial = F.normalize(cantings_e[:, :3], dim=1)  # (4, 3)
        canting_n_spatial = F.normalize(cantings_n[:, :3], dim=1)  # (4, 3)

        # Compute the third basis vector ("up") using the cross product.
        canting_u_spatial = torch.cross(canting_e_spatial, canting_n_spatial, dim=1)
        canting_u_spatial = F.normalize(canting_u_spatial, dim=1)  # (4, 3)

        # Build the 3x3 rotation matrix for each facet.
        # Each matrix has columns [canting_e_spatial, canting_n_spatial, canting_u_spatial].
        R_3x3 = torch.stack((canting_e_spatial, canting_n_spatial, canting_u_spatial), dim=2)  # (4, 3, 3)

        # Rotate the surface point cloud.
        # surface_pointcloud has shape (4, N, 3) so we can use batch matrix multiplication.
        rotated_surface_pointcloud = torch.bmm(surface_pointcloud, R_3x3.transpose(1, 2))
        
        # Translate the points.
        # facet_translation_vectors should be of shape (4, 3)
        translated_points = rotated_surface_pointcloud + facet_translation_vectors[:,:3].unsqueeze(1)
        
        # Split into a list (one tensor per facet)
        surface_points_with_facets_list = [translated_points[i] for i in range(translated_points.shape[0])]

        # Duplicate the surface normals (canting_u_spatial) for each point per facet.
        surface_normals_with_facets_list = [
            canting_u_spatial[i].unsqueeze(0).expand(translated_points.shape[1], -1)
            for i in range(canting_u_spatial.shape[0])
        ]

        analytical_cfg = AnalyticalConfig(
            facet_translation_vectors=facet_translation_vectors,
            canting_e=cantings_e,
            canting_n=cantings_n,
            facet_width=facet_width,
            facet_height=facet_height,
            device=device
)

        return surface_points_with_facets_list, surface_normals_with_facets_list, analytical_cfg

def load_measured_heliostat_pointcloud_from_paint_deflectometry_file(heliostat_file_path: str,deflectometry_file_path: str, device: Union[torch.device, str] = "cuda") -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    (
            number_of_facets,
            heliostat_height,
            heliostat_width,
            facet_translation_vectors,
            canting_e,
            canting_n,
        ) = loading_files_utils.load_heliostat_properties_file(
            heliostat_file_path=heliostat_file_path,
            device = device
        )
    deflectometry_result = _try_load_deflectometry(
        deflectometry_file_path, config_dictionary, number_of_facets, device
    )
    if deflectometry_result:
        surface_points_with_facets_list, surface_normals_with_facets_list = deflectometry_result
    else:
        raise ValueError(f"No Deflectometry data found at: {deflectometry_file_path}.")

    return surface_points_with_facets_list, surface_normals_with_facets_list, facet_translation_vectors, canting_e,canting_n