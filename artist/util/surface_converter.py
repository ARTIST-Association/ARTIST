import json
import logging
import pathlib
import struct
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt #TODO Remove Later

import torch
import torch.nn.functional as F

from artist.util import config_dictionary, loading_files_utils, utils
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

        
        params = [control_points.requires_grad_(True)]

        # Set up optimizer and scheduler
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
            optimizer.zero_grad(set_to_none=True)  # recommended for memory efficiency

            points, normals = nurbs_surface.calculate_surface_points_and_normals(
                device=device
            )

            # Compute loss
            if conversion_method == config_dictionary.convert_nurbs_from_points:
                loss = (points - surface_points).abs().mean()
            elif conversion_method == config_dictionary.convert_nurbs_from_normals:
                loss = (normals - surface_normals).abs().mean()
            else:
                raise NotImplementedError(
                    f"Conversion method {conversion_method} not yet implemented!"
                )

            loss.backward()
            if self.optimize_only_u:
                 with torch.no_grad():
                    for param_group in optimizer.param_groups:
                        for parameter in param_group["params"]: 
                            parameter.grad[...,0:2] = 0
 

            optimizer.step()
            scheduler.step(loss)

            if epoch % 100 == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}.",
                )
            epoch += 1

        return nurbs_surface


    def _generate_surface_config(
        self,
        surface_points_with_facets_list: list[torch.Tensor],
        surface_normals_with_facets_list: list[torch.Tensor],
        facet_translation_vectors: torch.Tensor,
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

    def generate_surface_config_from_point_cloud(
        self,
        surface_points_with_facets_list: list[torch.Tensor],
        surface_normals_with_facets_list: list[torch.Tensor],
        facet_translation_vectors: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> list[FacetConfig]:
        """
        Generate surface configuration from deflectometry and/or ideal data.

        Parameters
        ----------
        heliostat_file_path : pathlib.Path
            Path to PAINT heliostat JSON file.
        deflectometry_file_path : Optional[pathlib.Path]
            Path to deflectometry data.
        ideal_file : Optional[Union[pathlib.Path, torch.Tensor]]
            Ideal heliostat model.
        device : Union[torch.device, str]
            Device for tensor operations.

        Returns
        -------
        list[FacetConfig]
            Surface configuration using available measured/ideal data.
        """
        # Extract heliostat geometry
        

        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets_list,
            surface_normals_with_facets_list=surface_normals_with_facets_list,
            facet_translation_vectors=facet_translation_vectors,
            device=device,
        )
        return surface_config


def generate_ideal_surface_config_from_paint_heliostat_properties(
        self,
        heliostat_file_path: pathlib.Path,
        number_of_surface_points: int = 1000,
        device: Union[torch.device, str] = "cuda",
    ) -> list[FacetConfig]:
        """
        Generate surface configuration from heliostat geometry (no deflectometry or ideal data used).

        Parameters
        ----------
        heliostat_file_path : pathlib.Path
            Path to PAINT heliostat JSON file.
        device : Union[torch.device, str]
            Device for torch tensors.

        Returns
        -------
        list[FacetConfig]
            Surface configuration assuming ideal heliostat geometry.
        """
        number_of_facets, heliostat_height, heliostat_width, facet_translation_vectors, canting_e, canting_n = _load_heliostat_properties_file(
            heliostat_file_path=heliostat_file_path)

        surface_points_with_facets, surface_normals_with_facets = generate_ideal_juelich_heliostat_point_cloud(
            cantings_e=canting_e,
            cantings_n=canting_n,
            facet_translation_vectors=facet_translation_vectors,
            number_of_surface_points=1000,
            device=device
        )


        surface_config = self._generate_surface_config(
            surface_points_with_facets_list=surface_points_with_facets,
            surface_normals_with_facets_list=surface_normals_with_facets,
            heliostat_height=heliostat_height,
            heliostat_width=heliostat_width,
            facet_translation_vectors=facet_translation_vectors,
            canting_e=canting_e,
            canting_n=canting_n,
            device=device,
        )
        return surface_config


    
    






