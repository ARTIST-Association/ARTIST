import json
import logging
import pathlib
import struct

import h5py
import numpy as np
import torch

from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurface

log = logging.getLogger(__name__)
"""A logger for the surface converter."""


class SurfaceConverter:
    """
    Implement a converter that converts surface data from various sources to HDF5 format.

    Currently the surface converter can be used for ``STRAL`` or ``PAINT`` data.

    Attributes
    ----------
    number_of_evaluation_points : torch.Tensor
        The number of evaluation points in the east and north direction used to generate a discrete surface from NURBS.
    number_control_points : torch.Tensor
        Number of NURBS control points in the east an north direction.
    degrees : torch.Tensor
        Degree of the NURBS in the east and north direction. 
    step_size : int
        The size of the step used to reduce the number of evaluation points for compute efficiency.
    conversion_method : str
        The conversion method used to learn the NURBS.
    tolerance : float
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
        number_of_evaluation_points: torch.Tensor = torch.tensor([50, 50]),
        number_control_points: torch.Tensor = torch.tensor([10, 10]),
        degrees: torch.Tensor = torch.tensor([2, 2]),
        step_size: int = 100,
        conversion_method: str = config_dictionary.convert_nurbs_from_normals,
        tolerance: float = 3e-5,
        initial_learning_rate: float = 1e-3,
        max_epoch: int = 10000,
        device: torch.device | None = None,
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
        number_of_evaluation_points : torch.Tensor
            The number of evaluation points in the east and north direction used to generate a discrete surface from NURBS (default is torch.tensor([50, 50])).
        number_control_points : torch.Tensor
            Number of NURBS control points in the east an north direction (default is torch.tensor([10, 10])).
        degrees : torch.Tensor
            Degree of the NURBS in the east and north direction (default is torch.tensor([2, 2])). 
        step_size : int
            The size of the step used to reduce the number of evaluation points for compute efficiency (default is 100).
        conversion_method : str
            The conversion method used to learn the NURBS.
        tolerance : float
            Tolerance value used for fitting NURBS surfaces (default is 3e-5).
        initial_learning_rate : float
            Initial learning rate for the learning rate scheduler used when fitting NURBS surfaces (default is 1e-3).
        max_epoch : int
            Maximum number of epochs to use when fitting NURBS surfaces (default is 10000).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        self.number_of_evaluation_points = number_of_evaluation_points.to(device)
        self.number_control_points = number_control_points.to(device)
        self.degrees = degrees.to(device)
        self.step_size = step_size
        self.conversion_method = conversion_method
        self.tolerance = tolerance
        self.initial_learning_rate = initial_learning_rate
        self.max_epoch = max_epoch

    def fit_nurbs_surface(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: torch.device | None = None,
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
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        NURBSSurface
            A NURBS surface.
        """
        device = get_device(device=device)

        # Since NURBS are only defined between (0,1), we need to normalize the evaluation points and remove the boundary points.
        evaluation_points = surface_points.clone()
        evaluation_points[:, 0] = utils.normalize_points(evaluation_points[:, 0])
        evaluation_points[:, 1] = utils.normalize_points(evaluation_points[:, 1])
        evaluation_points[:, 2] = 0

        # Initialize the NURBS surface.
        control_points = torch.zeros((self.number_control_points[0], self.number_control_points[1], 3), device=device)

        #TODO: width and height always 1.0
        width_of_nurbs = torch.max(evaluation_points[:, 0]) - torch.min(
            evaluation_points[:, 0]
        )
        height_of_nurbs = torch.max(evaluation_points[:, 1]) - torch.min(
            evaluation_points[:, 1]
        )

        origin_offsets_e = torch.linspace(
            -width_of_nurbs / 2,
            width_of_nurbs / 2,
            self.number_control_points[0],
            device=device,
        )
        origin_offsets_n = torch.linspace(
            -height_of_nurbs / 2,
            height_of_nurbs / 2,
            self.number_control_points[1],
            device=device,
        )

        control_points_e, control_points_n = torch.meshgrid(origin_offsets_e, origin_offsets_n, indexing='ij')

        control_points[:, :, 0] = control_points_e
        control_points[:, :, 1] = control_points_n
        control_points[:, :, 2] = 0

        nurbs_surface = NURBSSurface(
            degrees=self.degrees,
            evaluation_points=evaluation_points,
            control_points=control_points,
            device=device,
        )

        # Optimize the control points of the NURBS surface.
        optimizer = torch.optim.Adam(
            nurbs_surface.parameters(), lr=self.initial_learning_rate
        )
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
        while loss > self.tolerance and epoch <= self.max_epoch:
            points, normals = nurbs_surface.calculate_surface_points_and_normals(
                device=device
            )

            optimizer.zero_grad()

            if self.conversion_method == config_dictionary.convert_nurbs_from_points:
                loss = (points - surface_points).abs().mean()
            elif self.conversion_method == config_dictionary.convert_nurbs_from_normals:
                loss = (normals - surface_normals).abs().mean()
            else:
                raise NotImplementedError(
                    f"Conversion method {self.conversion_method} not yet implemented!"
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
        facet_translation_vectors: torch.Tensor,
        canting: torch.Tensor,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a surface configuration from a data source.

        Parameters
        ----------
        surface_points_with_facets_list : list[torch.Tensor]
            A list of facetted surface points. Points per facet may vary.
        surface_normals_with_facets_list : list[torch.Tensor]
            A list of facetted surface normals. Normals per facet may vary.
        facet_translation_vectors : torch.Tensor
            Translation vector for each facet from heliostat origin to relative position.
        canting : torch.Tensor
            The canting vector per facet in east and north direction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        SurfaceConfig
            A surface configuration.
        """
        device = get_device(device=device)

        log.info("Beginning generation of the surface configuration based on data.")

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
        canting = utils.convert_3d_direction_to_4d_format(canting, device=device)
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
                device=device,
            )
            facet_config_list.append(
                FacetConfig(
                    facet_key=f"facet_{i + 1}",
                    control_points=nurbs_surface.control_points.detach(),
                    degrees=nurbs_surface.degrees,
                    number_of_evaluation_points=self.number_of_evaluation_points,
                    translation_vector=facet_translation_vectors[i],
                    canting=canting[i],
                )
            )
        
        surface_config = SurfaceConfig(facet_list=facet_config_list)
        log.info("Surface configuration based on data complete!")
        return surface_config

    def generate_surface_config_from_stral(
        self,
        stral_file_path: pathlib.Path,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a surface configuration from a ``STRAL`` file.

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
        SurfaceConfig
            A surface configuration.
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

        log.info("Loading ``STRAL`` data complete.")

        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
            facet_translation_vectors=facet_translation_vectors,
            canting=canting,
            device=device,
        )

        return surface_config

    def generate_surface_config_from_paint(
        self,
        heliostat_file_path: pathlib.Path,
        deflectometry_file_path: pathlib.Path,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a surface configuration from a ``PAINT`` dataset.

        Parameters
        ----------
        deflectometry_file_path : pathlib.Path
            The file path to the ``PAINT`` deflectometry data that will be converted.
        heliostat_file_path : pathlib.Path
            The file path to the ``PAINT`` heliostat properties data that will be converted.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        SurfaceConfig
            A surface configuration.
        """
        device = get_device(device=device)

        log.info("Beginning extraction of data from ```PAINT``` file.")
        # Reading ``PAINT`` heliostat json file.
        with open(heliostat_file_path, "r") as file:
            heliostat_dict = json.load(file)
            number_of_facets = heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_number_of_facets
            ]

            facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
            canting = torch.empty(number_of_facets, 2, 3, device=device)

            for facet in range(number_of_facets):
                facet_translation_vectors[facet, :] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_translation_vector],
                    device=device,
                )
                canting[facet, 0] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_canting_e],
                    device=device,
                )
                canting[facet, 1] = torch.tensor(
                    heliostat_dict[config_dictionary.paint_facet_properties][
                        config_dictionary.paint_facets
                    ][facet][config_dictionary.paint_canting_n],
                    device=device,
                )

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

        log.info("Loading ``PAINT`` data complete.")

        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
            facet_translation_vectors=facet_translation_vectors,
            canting=canting,
            device=device,
        )

        return surface_config


    def generate_ideal_surface_config_from_paint(
        self,
        heliostat_file_path: pathlib.Path,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a surface configuration from a ``PAINT`` dataset.

        Parameters
        ----------
        heliostat_file_path : pathlib.Path
            The file path to the ``PAINT`` heliostat properties data that will be converted.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        SurfaceConfig
            A surface configuration.
        """
        device = get_device(device=device)

        log.info("Beginning extraction of data from ```PAINT``` file.")
        # Reading ``PAINT`` heliostat json file.
        with open(heliostat_file_path, "r") as file:
            heliostat_dict = json.load(file)
        
        number_of_facets = heliostat_dict[config_dictionary.paint_facet_properties][
            config_dictionary.paint_number_of_facets
        ]

        facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
        canting = torch.empty(number_of_facets, 2, 3, device=device)

        for facet in range(number_of_facets):
            facet_translation_vectors[facet, :] = torch.tensor(
                heliostat_dict[config_dictionary.paint_facet_properties][
                    config_dictionary.paint_facets
                ][facet][config_dictionary.paint_translation_vector],
                device=device,
            )
            canting[facet, 0, :] = torch.tensor(
                heliostat_dict[config_dictionary.paint_facet_properties][
                    config_dictionary.paint_facets
                ][facet][config_dictionary.paint_canting_e],
                device=device,
            )
            canting[facet, 1, :] = torch.tensor(
                heliostat_dict[config_dictionary.paint_facet_properties][
                    config_dictionary.paint_facets
                ][facet][config_dictionary.paint_canting_n],
                device=device,
            )

        log.info(
            "Initializing ideal heliostat surface."
        )
        facet_config_list = []

        control_points = torch.zeros((self.number_control_points[0], self.number_control_points[1], 3), device=device)
        origin_offsets_e = torch.linspace(
            -0.5, 0.5,
            control_points.shape[0],
            device=device,
        )
        origin_offsets_n = torch.linspace(
            -0.5, 0.5,
            control_points.shape[1],
            device=device,
        )
        
        control_points_e, control_points_n = torch.meshgrid(origin_offsets_e, origin_offsets_n, indexing='ij')

        control_points[:, :, 0] = control_points_e
        control_points[:, :, 1] = control_points_n
        control_points[:, :, 2] = 0

        for facet_index in range(number_of_facets):
            facet_config = FacetConfig(
                facet_key=f"facet_{facet_index + 1}",
                control_points=control_points,
                degrees=self.degrees,
                number_evaluation_points=self.number_of_evaluation_points,
                translation_vector=facet_translation_vectors[facet_index],
                canting=canting[facet_index],
            )
            facet_config_list.append(facet_config)
        
        surface_config = SurfaceConfig(facet_list=facet_config_list)
        log.info("Surface configuration based on ideal heliostat complete!")
        
        return surface_config
