import json
import logging
import pathlib
import struct
from typing import Optional, Union

import matplotlib.pyplot as plt #TODO Remove Later

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from artist.util import config_dictionary, utils
from artist.util.configuration_classes import FacetConfig
from artist.util.nurbs import NURBSSurface

log = logging.getLogger(__name__)
"""A logger for the surface converter."""


class SurfaceConverter:
    """
    Implement a converter that converts surface data from various sources to HDF5 format.

    Currently the surface converter can be used for ``STRAL`` or ``PAINT`` data.

    Attributes
    ----------
    step_size : int
        The size of the step used to reduce the number of considered points for compute efficiency.
    number_eval_points_e : int
        The number of evaluation points in the east direction used to generate a discrete surface from NURBS.
    number_eval_points_n : int
        The number of evaluation points in the north direction used to generate a discrete surface from NURBS.
    conversion_method : str
        The conversion method used to learn the NURBS.
    number_control_points_e : int
        Number of NURBS control points in the east direction.
    number_control_points_n : int
        Number of NURBS control points in the north direction.
    degree_e : int
        Degree of the NURBS in the east (first) direction.
    degree_n : int
        Degree of the NURBS in the north (second) direction.
    tolerance : float, optional
        Tolerance value used for fitting NURBS surfaces.
    initial_learning_rate : float
        Initial learning rate for the learning rate scheduler used when fitting NURBS surfaces.
    max_epoch : int
        Maximum number of epochs to use when fitting NURBS surfaces.

    Methods
    -------
    fit_nurbs_surface()
        Fit the NURBS surface given the conversion method.
    generate_surface_config_from_stral()
        Generate a surface configuration from a ``STRAL`` file.
    generate_surface_config_from_paint()
        Generate a surface configuration from a ``PAINT`` dataset.
    """

    def __init__(
        self,
        step_size: int = 100,
        number_eval_points_e: int = 50,
        number_eval_points_n: int = 50,
        conversion_method: str = config_dictionary.convert_nurbs_from_normals,
        number_control_points_e: int = 20,
        number_control_points_n: int = 20,
        degree_e: int = 3,
        degree_n: int = 3,
        tolerance: float = 3e-5,
        initial_learning_rate: float = 1e-3,
        max_epoch: int = 10000,
        optimize_only_u: bool = True
    ) -> None:
        """
        Initialize the converter.

        Heliostat data, including information regarding their surfaces and structure, can be generated via ``STRAL`` and
        exported to a binary file or downloaded from ```PAINT``. The data formats are different depending on their source.
        To convert this data into a surface configuration format suitable for ``ARTIST``, this converter first loads the
        data and then learns NURBS surfaces based on the data. Finally, the converter returns a list of facets that can
        be used directly in an ``ARTIST`` scenario.

        Parameters
        ----------
        step_size : int
            The size of the step used to reduce the number of considered points for compute efficiency (default is 100).
        number_eval_points_e : int
            The number of evaluation points in the east direction used to generate a discrete surface from NURBS.
        number_eval_points_n : int
            The number of evaluation points in the north direction used to generate a discrete surface from NURBS.
        conversion_method : str
            The conversion method used to learn the NURBS.
        number_control_points_e : int
            Number of NURBS control points in the east direction (default: 10).
        number_control_points_n : int
            Number of NURBS control points in the north direction (default: 10).
        degree_e : int
            Degree of the NURBS in the east (first) direction (default: 2).
        degree_n : int
            Degree of the NURBS in the north (second) direction (default: 2).
        tolerance : float, optional
            Tolerance value used for fitting NURBS surfaces (default: 1e-5).
        initial_learning_rate : float
            Initial learning rate for the learning rate scheduler used when fitting NURBS surfaces (default: 1e-1).
        max_epoch : int
            Maximum number of epochs to use when fitting NURBS surfaces (default: 10000).

        """
        self.step_size = step_size

        self.number_eval_points_e = number_eval_points_e
        self.number_eval_points_n = number_eval_points_n
        self.conversion_method = conversion_method
        self.number_control_points_e = number_control_points_e
        self.number_control_points_n = number_control_points_n
        self.degree_e = degree_e
        self.degree_n = degree_n
        self.tolerance = tolerance
        self.initial_learning_rate = initial_learning_rate
        self.max_epoch = max_epoch
        self.optimize_only_u = optimize_only_u
        if not optimize_only_u:
            log.warning("Warning: Optimizing E/N/U may cause errors in canting representation.")

    def fit_nurbs_surface(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        conversion_method: str,
        number_control_points_e: int = 10,
        number_control_points_n: int = 10,
        degree_e: int = 2,
        degree_n: int = 2,
        tolerance: float = 1e-5,
        initial_learning_rate: float = 1e-1,
        max_epoch: int = 2500,
        device: Union[torch.device, str] = "cuda",
    ) -> NURBSSurface:
        """
        Fit the NURBS surface given the conversion method.

        The surface points are first normalized and shifted to the range (0,1) to be compatible with the knot vector of
        the NURBS surface. The NURBS surface is then initialized with the correct number of control points, degrees, and
        knots, and the origin of the control points is set based on the width and height of the point cloud. The control
        points are then fitted to the surface points or surface normals using an Adam optimizer.
        The optimization stops when the loss is less than the tolerance or the maximum number of epochs is reached.

        Parameters
        ----------
        surface_points : torch.Tensor
            The surface points given as an (N, 4) tensor.
        surface_normals : torch.Tensor
            The surface normals given as an (N, 4) tensor.
        conversion_method : str
            The conversion method used to learn the NURBS.
        number_control_points_e : int
            Number of NURBS control points to be set in the east (first) direction (default: 10).
        number_control_points_n : int
            Number of NURBS control points to be set in the north (second) direction (default: 10).
        degree_e : int
            Degree of the NURBS in the east (first) direction (default: 2).
        degree_n : int
            Degree of the NURBS in the north (second) direction (default: 2).
        tolerance : float, optional
            Tolerance value for convergence criteria (default: 1e-5).
        initial_learning_rate : float
            Initial learning rate for the learning rate scheduler (default: 1e-1).
        max_epoch : int, optional
            Maximum number of epochs for optimization (default: 2500).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        NURBSSurface
            A NURBS surface.
        """
        device = torch.device(device)
        # Since NURBS are only defined between (0,1), we need to normalize the evaluation points and remove the boundary points.
        evaluation_points = surface_points.clone()
        evaluation_points[:, 2] = 0
        evaluation_points_e = utils.normalize_points(evaluation_points[:, 0])
        evaluation_points_n = utils.normalize_points(evaluation_points[:, 1])

        # Initialize the NURBS surface.
        control_points_shape = (number_control_points_e, number_control_points_n)
        control_points = torch.zeros(control_points_shape + (3,), device=device)
        width_of_nurbs = torch.max(evaluation_points[:, 0]) - torch.min(
            evaluation_points[:, 0]
        )
        height_of_nurbs = torch.max(evaluation_points[:, 1]) - torch.min(
            evaluation_points[:, 1]
        )
        origin_offsets_e = torch.linspace(
            -width_of_nurbs / 2,
            width_of_nurbs / 2,
            number_control_points_e,
            device=device,
        )
        origin_offsets_n = torch.linspace(
            -height_of_nurbs / 2,
            height_of_nurbs / 2,
            number_control_points_n,
            device=device,
        )
        origin_offsets = torch.cartesian_prod(origin_offsets_e, origin_offsets_n)
        origin_offsets = torch.hstack(
            (
                origin_offsets,
                torch.zeros((len(origin_offsets), 1), device=device),
            )
        )
        control_points = torch.nn.parameter.Parameter(
            origin_offsets.reshape(control_points.shape)
        )
        nurbs_surface = NURBSSurface(
            degree_e,
            degree_n,
            evaluation_points_e,
            evaluation_points_n,
            control_points,
            device=device,
        )
        if self.optimize_only_u:
            control_points = control_points.detach()
            optimizable_control_points = control_points[..., 2]
            optimizable_control_points.requires_grad_(True)

            # For optimization we will only update uu and reconstruct control_points during loss computation
            params = [optimizable_control_points]
        else:
            optimizable_control_points = control_points
            optimizable_control_points.requires_grad_(True)
            params = [optimizable_control_points]



        # Optimize the control points of the NURBS surface.
        optimizer = torch.optim.Adam(params, lr=initial_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=50,
            threshold=1e-7,
            threshold_mode="abs",
        )
        loss = torch.inf
        epoch = 0
        while loss > tolerance and epoch <= max_epoch:

            if self.optimize_only_u:
                # Reconstruct control points
                updated_control_points = torch.stack([
                    control_points[..., 0],  # fixed E
                    control_points[..., 1],  # fixed N
                    optimizable_control_points  # optimized U
                ], dim=-1)
            else:
                updated_control_points = optimizable_control_points

            nurbs_surface.control_points = updated_control_points

            points, normals = nurbs_surface.calculate_surface_points_and_normals(
                device=device
            )

            optimizer.zero_grad()

            if conversion_method == config_dictionary.convert_nurbs_from_points:
                loss = (points - surface_points).abs().mean()
            elif conversion_method == config_dictionary.convert_nurbs_from_normals:
                loss = (normals - surface_normals).abs().mean()
            else:
                raise NotImplementedError(
                    f"Conversion method {conversion_method} not yet implemented!"
                )
            loss.backward()

            optimizer.step()
            scheduler.step(loss.abs().mean())
            if epoch % 100 == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}.",
                )
            epoch += 1

        return nurbs_surface

    def _generate_surface_config(
        self,
        surface_points_with_facets_list: list[torch.Tensor],
        surface_normals_with_facets_list: list[torch.Tensor],
        heliostat_height: float,
        heliostat_width: float,
        facet_translation_vectors: torch.Tensor,
        canting_e: torch.Tensor,
        canting_n: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> list[FacetConfig]:
        """
        Generate a surface configuration from a data source.

        Parameters
        ----------
        surface_points_with_facets_list : list[torch.Tensor]
            A list of facetted surface points. Points per facet may vary.
        surface_normals_with_facets_list : list[torch.Tensor]
            A list of facetted surface normals. Normals per facet may vary.
        heliostat_height : float
            The height of the heliostat.
        heliostat_width : float
            The width of the heliostat.
        facet_translation_vectors : torch.Tensor
            Translation vector for each facet from heliostat origin to relative position.
        canting_e : torch.Tensor
            The canting vector per facet in east direction.
        canting_n : torch.Tensor
            The canting vector per facet in north direction.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        list[FacetConfig]
            A list of facet configurations used to generate a surface.
        """
        log.info("Beginning generation of the surface configuration based on data.")
        device = torch.device(device)

        # All single_facet_surface_points and single_facet_surface_normals must have the same
        # dimensions, so that they can be stacked into a single tensor and then can be used by artist.
        min_x = min(
            single_facet_surface_points.shape[0]
            for single_facet_surface_points in surface_points_with_facets_list
        )
        reduced_single_facet_surface_points = [
            single_facet_surface_points[:min_x]
            for single_facet_surface_points in surface_points_with_facets_list
        ]
        surface_points_with_facets = torch.stack(reduced_single_facet_surface_points)

        min_x = min(
            single_facet_surface_normals.shape[0]
            for single_facet_surface_normals in surface_normals_with_facets_list
        )
        reduced_single_facet_surface_normals = [
            single_facet_surface_normals[:min_x]
            for single_facet_surface_normals in surface_normals_with_facets_list
        ]
        surface_normals_with_facets = torch.stack(reduced_single_facet_surface_normals)

        # Select only selected number of points to reduce compute.
        surface_points_with_facets = surface_points_with_facets[:, :: self.step_size]
        surface_normals_with_facets = surface_normals_with_facets[:, :: self.step_size]

        # Convert to 4D format.
        facet_translation_vectors = utils.convert_3d_direction_to_4d_format(
            facet_translation_vectors, device=device
        )
        # If we are using a point cloud to learn the points, we do not need to translate the facets.
        if self.conversion_method == config_dictionary.convert_nurbs_from_points:
            facet_translation_vectors = torch.zeros(
                facet_translation_vectors.shape, device=device
            )
        # Convert to 4D format.
        canting_n = utils.convert_3d_direction_to_4d_format(canting_n, device=device)
        canting_e = utils.convert_3d_direction_to_4d_format(canting_e, device=device)
        surface_points_with_facets = utils.convert_3d_point_to_4d_format(
            surface_points_with_facets, device=device
        )
        surface_normals_with_facets = utils.convert_3d_direction_to_4d_format(
            surface_normals_with_facets, device=device
        )

        # Convert to NURBS surface.
        log.info("Converting to NURBS surface.")
        facet_config_list = []
        for i in range(surface_points_with_facets.shape[0]):
            log.info(
                f"Converting facet {i + 1} of {surface_points_with_facets.shape[0]}."
            )
            nurbs_surface = self.fit_nurbs_surface(
                surface_points=surface_points_with_facets[i],
                surface_normals=surface_normals_with_facets[i],
                conversion_method=self.conversion_method,
                number_control_points_e=self.number_control_points_e,
                number_control_points_n=self.number_control_points_n,
                degree_e=self.degree_e,
                degree_n=self.degree_n,
                tolerance=self.tolerance,
                initial_learning_rate=self.initial_learning_rate,
                max_epoch=self.max_epoch,
                device=device,
            )
            facet_config_list.append(
                FacetConfig(
                    facet_key=f"facet_{i + 1}",
                    control_points=nurbs_surface.control_points.detach(),
                    degree_e=nurbs_surface.degree_e,
                    degree_n=nurbs_surface.degree_n,
                    number_eval_points_e=self.number_eval_points_e,
                    number_eval_points_n=self.number_eval_points_n,
                    translation_vector=facet_translation_vectors[i],
                    canting_e=canting_e[i],
                    canting_n=canting_n[i],
                )
            )
        log.info("Surface configuration based on data complete!")
        return facet_config_list

    def generate_surface_config_from_stral(
        self,
        stral_file_path: pathlib.Path,
        device: Union[torch.device, str] = "cuda",
    ) -> list[FacetConfig]:
        """
        Generate a surface configuration from a ``STRAL`` file.

        Parameters
        ----------
        stral_file_path : pathlib.Path
            The file path to the ``STRAL`` data that will be converted.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        list[FacetConfig]
            A list of facet configurations used to generate a surface.
        """
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

        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
            heliostat_height=height,
            heliostat_width=width,
            facet_translation_vectors=facet_translation_vectors,
            canting_e=canting_e,
            canting_n=canting_n,
            device=device,
        )

        return surface_config

    def generate_ideal_juelich_heliostat_surface(
        self,
        cantings_e: torch.Tensor,
        cantings_n: torch.Tensor,
        facet_translation_vectors: torch.Tensor,
        number_of_surface_points: int,
        device: Union[torch.device, str] = "cuda"
        ) -> torch.Tensor:
        """
        Generate an ideal Jülich heliostat surface as a collection of points on 4 facets.
        """

        ###CANTING ADAPTION REMOVE LATER####
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

        return surface_points_with_facets_list, surface_normals_with_facets_list
    
    def generate_surface_config_from_paint(
        self,
        heliostat_file_path: pathlib.Path,
        deflectometry_file_path: Optional[pathlib.Path] = None,
        number_of_surface_points_for_ideal_surface: Optional[int]= 2000,
        device: Union[torch.device, str] = "cuda",
    ) -> list[FacetConfig]:
        """
        Generate a surface configuration from a ``PAINT`` dataset.

        If no deflectometry file path is specified, TODO 

        Parameters
        ----------
        deflectometry_file_path : Optional[pathlib.Path]
            The file path to the ``PAINT`` deflectometry data that will be converted (default is None).
        heliostat_file_path : pathlib.Path
            The file path to the ``PAINT`` heliostat properties data that will be converted.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        list[FacetConfig]
            A list of facet configurations used to generate a surface.
        """
        log.info("Beginning extraction of data from ```PAINT``` file.")
        # Reading ``PAINT`` heliostat json file.
        with open(heliostat_file_path, "r") as file:
            heliostat_dict = json.load(file)
            number_of_facets = heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_number_of_facets
            ]

            heliostat_height = heliostat_dict[config_dictionary.paint_heliostat_height]
            heliostat_width = heliostat_dict[config_dictionary.paint_heliostat_width]

            facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
            canting_e = torch.empty(number_of_facets, 3, device=device)
            canting_n = torch.empty(number_of_facets, 3, device=device)

            for facet in range(number_of_facets):
                facet_translation_vectors[facet, :] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_translation_vetor],
                    device=device,
                )
                canting_e[facet, :] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_canting_e],
                    device=device,
                )
                canting_n[facet, :] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_canting_n],
                    device=device,
                ) 

        
        if deflectometry_file_path is not None and pathlib.Path(deflectometry_file_path).is_file():
            # Reading ``PAINT`` deflectometry hdf5 file.
            log.info(
                f"Reading PAINT deflectometry file located at: {deflectometry_file_path}."
            )
            with h5py.File(deflectometry_file_path, "r") as file:
                surface_points_with_facets_list = []
                surface_normals_with_facets_list = []
                for f in range(number_of_facets):
                    number_of_points = len(
                        file[f"{config_dictionary.paint_facet}{f + 1}"][
                            config_dictionary.paint_surface_points
                        ]
                    )
                    single_facet_surface_points = torch.empty(
                        number_of_points, 3, device=device
                    )
                    single_facet_surface_normals = torch.empty(
                        number_of_points, 3, device=device
                    )

                    points_data = torch.tensor(
                        np.array(
                            file[f"{config_dictionary.paint_facet}{f + 1}"][
                                config_dictionary.paint_surface_points
                            ]
                        ),
                        device=device,
                    )
                    normals_data = torch.tensor(
                        np.array(
                            file[f"{config_dictionary.paint_facet}{f + 1}"][
                                config_dictionary.paint_surface_normals
                            ]
                        ),
                        device=device,
                    )

                    for i, point_data in enumerate(points_data):
                        single_facet_surface_points[i, :] = point_data
                    for i, normal_data in enumerate(normals_data):
                        single_facet_surface_normals[i, :] = normal_data
                    surface_points_with_facets_list.append(single_facet_surface_points)
                    surface_normals_with_facets_list.append(single_facet_surface_normals)

            log.info("Loading ``PAINT`` with defletometry data complete.")
        else:
            log.info(
            f"Deflectometry not found or is not a deflectometry file at location: {deflectometry_file_path}."
            )
            surface_points_with_facets_list, surface_normals_with_facets_list = self.generate_ideal_juelich_heliostat_surface(
            cantings_e = canting_e,
            cantings_n = canting_n,
            facet_translation_vectors = facet_translation_vectors,
            number_of_surface_points=number_of_surface_points_for_ideal_surface,
            device = device
            )
            
            
            log.info("Loading ``PAINT`` without defletometry data complete. Created an ideal heliostat.")

        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
            heliostat_height=heliostat_height,
            heliostat_width=heliostat_width,
            facet_translation_vectors=facet_translation_vectors,
            canting_e=canting_e,
            canting_n=canting_n,
            device=device,
        )

        return surface_config
    
    

def compute_grid_dimensions(facet_width: float, facet_height: float, points_per_facet: float) -> tuple[int, int]:
    """
    Compute the number of columns and rows for a grid that maintains an aspect ratio.

    Parameters:
    ----------
    facet_width : float
        Width of the facet.
    facet_height : float
        Height of the facet.
    points_per_facet : float
        Approximate number of points for each facet.

    Returns:
    -------
    int, int
        Number of columns and rows for the grid.
    """
    aspect_ratio = facet_width / facet_height

    # Convert to tensor to ensure compatibility with torch operations
    points_per_facet_tensor = torch.tensor(points_per_facet, dtype=torch.float32)
    aspect_ratio_tensor = torch.tensor(aspect_ratio, dtype=torch.float32)

    # Compute an approximate number of columns
    number_of_columns = torch.round(torch.sqrt(points_per_facet_tensor * aspect_ratio_tensor)).int().item()

    # Compute rows such that the total points approximate `points_per_facet`
    number_of_rows = torch.round(points_per_facet_tensor / number_of_columns).int().item()

    return number_of_columns, number_of_rows

def plot_pointcloud(pointcloud: torch.Tensor) -> None:
    """
    Plot a pointcloud generated by generate_ideal_juelich_heliostat_surface.
    
    The pointcloud is expected to be of shape (number_of_facets, number_of_surface_points, 3),
    where the coordinates are in the East-North-Up (ENU) coordinate system.
    """
    # Convert the pointcloud to a numpy array (moving to CPU if necessary)
    pc_np = pointcloud.detach().cpu().numpy()
    
    # Create a new figure and axis for the plot.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define a set of colors for different facets.
    colors = ['red', 'blue', 'green', 'purple']
    
    # Iterate over the facets and plot each one.
    for i in range(pc_np.shape[0]):
        facet = pc_np[i]  # shape: (number_of_surface_points, 3)
        east = facet[:, 0]
        north = facet[:, 1]
        ax.scatter(east, north, color=colors[i % len(colors)], label=f'Facet {i+1}', s=20)
    
    # Label the axes and set plot title.
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_title("Heliostat Surface Point Cloud (East-North Plane)")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")  # Ensures equal scaling on both axes.
    
    plt.savefig("pointcloud.png")
    plt.close(fig)


import torch
import math

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
    
    

def compute_grid_dimensions(facet_width: float, facet_height: float, points_per_facet: float) -> tuple[int, int]:
    """
    Compute the number of columns and rows for a grid that maintains an aspect ratio.

    Parameters:
    ----------
    facet_width : float
        Width of the facet.
    facet_height : float
        Height of the facet.
    points_per_facet : float
        Approximate number of points for each facet.

    Returns:
    -------
    int, int
        Number of columns and rows for the grid.
    """
    aspect_ratio = facet_width / facet_height

    # Convert to tensor to ensure compatibility with torch operations
    points_per_facet_tensor = torch.tensor(points_per_facet, dtype=torch.float32)
    aspect_ratio_tensor = torch.tensor(aspect_ratio, dtype=torch.float32)

    # Compute an approximate number of columns
    number_of_columns = torch.round(torch.sqrt(points_per_facet_tensor * aspect_ratio_tensor)).int().item()

    # Compute rows such that the total points approximate `points_per_facet`
    number_of_rows = torch.round(points_per_facet_tensor / number_of_columns).int().item()

    return number_of_columns, number_of_rows

def plot_pointcloud(pointcloud: torch.Tensor) -> None:
    """
    Plot a pointcloud generated by generate_ideal_juelich_heliostat_surface.
    
    The pointcloud is expected to be of shape (number_of_facets, number_of_surface_points, 3),
    where the coordinates are in the East-North-Up (ENU) coordinate system.
    """
    # Convert the pointcloud to a numpy array (moving to CPU if necessary)
    pc_np = pointcloud.detach().cpu().numpy()
    
    # Create a new figure and axis for the plot.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define a set of colors for different facets.
    colors = ['red', 'blue', 'green', 'purple']
    
    # Iterate over the facets and plot each one.
    for i in range(pc_np.shape[0]):
        facet = pc_np[i]  # shape: (number_of_surface_points, 3)
        east = facet[:, 0]
        north = facet[:, 1]
        ax.scatter(east, north, color=colors[i % len(colors)], label=f'Facet {i+1}', s=20)
    
    # Label the axes and set plot title.
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_title("Heliostat Surface Point Cloud (East-North Plane)")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")  # Ensures equal scaling on both axes.
    
    plt.savefig("pointcloud.png")
    plt.close(fig)


import torch
import math

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
        print(f"Using {actual_point_count} points, which is {point_difference} more than requested {desired_points_per_facet}.")
    elif point_difference < 0:
        print(f"Using {actual_point_count} points, which is {-point_difference} less than requested {desired_points_per_facet}.")
    else:
        print(f"Using exactly {actual_point_count} points as requested.")

    # --- Step 4: Repeat for all facets ---
    complete_point_cloud = facet_point_cloud.unsqueeze(0).repeat(number_facets, 1, 1)
    return complete_point_cloud







#Facet Spans (East): [[-0.6374922394752502, 1.9569215510273352e-05, 0.0031505227088928223], [-0.6374922394752502, -1.9569215510273352e-05, 0.0031505227088928223], [-0.6374922394752502, -1.9569215510273352e-05, -0.0031505227088928223], [-0.6374922394752502, 1.9569215510273352e-
#Facet Spans (North): [[-0.0, 0.8024845123291016, -0.004984567407518625], [-0.0, 0.8024845123291016, 0.004984567407518625], [-0.0, 0.8024845123291016, -0.004984567407518625], [-0.0, 0.8024845123291016, 0.004984567407518625]]
