import json
import logging
import pathlib
from typing import List, Union

import h5py
import numpy
import torch

from artist.util import config_dictionary, utils
from artist.util.configuration_classes import FacetConfig
from artist.util.nurbs import NURBSSurface

log = logging.getLogger(__name__)
"""A logger for the ``PAINT`` to surface converter."""


class PAINTToSurfaceConverter:
    """
    Implement a converter that converts ``PAINT`` data to HDF5 format.

    Attributes
    ----------
    deflectometry_file_path : Path
        The file path to the ``PAINT`` deflectometry h5 data file that will be converted.
    heliostat_file_path : Path
        The file path to the ``PAINT`` heliostat json file that will be converted.
    step_size : int
        The size of the step used to reduce the number of considered points for compute efficiency.

    Methods
    -------
    normalize_evaluation_points_for_nurbs()
        Normalize evaluation points for NURBS with minimum > 0 and maximum < 1.
    fit_nurbs_surface()
        Fit the nurbs surface given the conversion method.
    generate_surface_config_from_paint()
        Generate a surface configuration based on the ``PAINT`` data.
    """

    def __init__(
        self,
        deflectometry_file_path: pathlib.Path,
        heliostat_file_path: pathlib.Path,
        step_size: int,
    ) -> None:
        """
        Initialize the converter.

        Deflectometry data is provided in hdf5 files by ``PAINT```. To convert this data into a surface configuration 
        format suitable for ``ARTIST``, this converter first loads the data and then learns NURBS surfaces based on
        the data. Finally, the converter returns a list of facets that can be used directly in an ``ARTIST`` scenario.

        Parameters
        ----------
        deflectometry_file_path : pathlib.Path
            The file path to the ``PAINT`` deflectometry h5 data file that will be converted.
        heliostat_file_path : pathlib.Path
            The file path to the ``PAINT`` heliostat json file that will be converted.
        step_size : int
            The size of the step used to reduce the number of considered points for compute efficiency.
        """
        self.deflectometry_file_path = deflectometry_file_path
        self.heliostat_file_path = heliostat_file_path
        self.step_size = step_size

    @staticmethod
    def normalize_evaluation_points_for_nurbs(evaluation_points: torch.Tensor) -> torch.Tensor:
        """
        Normalize the evaluation points for NURBS.

        This function normalizes the evaluation points for NURBS in the open interval of (0,1) since NURBS are not
        defined for the edges.

        Parameters
        ----------
        points : torch.Tensor
            The evaluation points for NURBS.

        Returns
        -------
        torch.Tensor
            The normalized evaluation points for NURBS.
        """
        # Since NURBS are only defined between (0,1), a small offset is required to exclude the boundaries from the
        # defined evaluation points.
        evaluation_points_normalized = (evaluation_points[:] - min(evaluation_points[:]) + 1e-5) / max(
            (evaluation_points[:] - min(evaluation_points[:])) + 2e-5
        )
        return evaluation_points_normalized

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
        Generate a NURBS surface based on ``STRAL`` data.

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

        # Normalize evaluation points and shift them so that they correspond to the knots.
        evaluation_points = surface_points.clone()
        evaluation_points[:, 2] = 0
        evaluation_points_e = self.normalize_evaluation_points_for_nurbs(
            evaluation_points[:, 0]
        )
        evaluation_points_n = self.normalize_evaluation_points_for_nurbs(
            evaluation_points[:, 1]
        )

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

        # Optimize the control points of the NURBS surface.
        optimizer = torch.optim.Adam([control_points], lr=initial_learning_rate)
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
                    f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}",
                )
            epoch += 1

        return nurbs_surface

    def generate_surface_config_from_paint(
        self,
        number_eval_points_e: int,
        number_eval_points_n: int,
        conversion_method: str,
        number_control_points_e: int = 10,
        number_control_points_n: int = 10,
        degree_e: int = 2,
        degree_n: int = 2,
        tolerance: float = 1e-5,
        initial_learning_rate: float = 1e-1,
        max_epoch: int = 10000,
        device: Union[torch.device, str] = "cuda",
    ) -> List[FacetConfig]:
        """
        Generate a surface configuration from a ``PAINT`` dataset.

        Parameters
        ----------
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
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        List[FacetConfig]
            A list of facet configurations used to generate a surface.
        """
        log.info(
            "Beginning generation of the surface configuration based on PAINT data."
        )
        with open(self.heliostat_file_path, "r") as file:
            heliostat_dict = json.load(file)
            width = heliostat_dict["width"]
            height = heliostat_dict["height"]
            number_of_facets = heliostat_dict["facet_properties"]["number_of_facets"]

            facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
            canting_e = torch.empty(number_of_facets, 3, device=device)
            canting_n = torch.empty(number_of_facets, 3, device=device)

            for facet in range(number_of_facets):
                facet_translation_vectors[facet, :] = torch.tensor(
                    heliostat_dict["facet_properties"]["facets"][facet][
                        "translation_vector"
                    ],
                    device=device,
                )
                canting_e[facet, :] = torch.tensor(
                    heliostat_dict["facet_properties"]["facets"][facet]["canting_e"],
                    device=device,
                )
                canting_n[facet, :] = torch.tensor(
                    heliostat_dict["facet_properties"]["facets"][facet]["canting_n"],
                    device=device,
                )

        # Reading ``PAINT`` deflectometry hdf5 file.
        # TODO
        # How to deal with different amount of points on different facets?
        log.info(
            f"Reading PAINT deflectometry file located at: {self.deflectometry_file_path}"
        )
        with h5py.File(self.deflectometry_file_path, "r") as file:
            surface_points_with_facets = torch.empty(0, device=device)
            surface_normals_with_facets = torch.empty(0, device=device)
            for f in range(number_of_facets):
                number_of_points = len(file[f"facet{f+1}"]["surface_points"])
                if f == 0:
                    surface_points_with_facets = torch.empty(
                        number_of_facets, number_of_points, 3, device=device
                    )
                    surface_normals_with_facets = torch.empty(
                        number_of_facets, number_of_points, 3, device=device
                    )
                points_data = torch.tensor(
                    numpy.array(file[f"facet{f+1}"]["surface_points"]), device=device
                )
                normals_data = torch.tensor(
                    numpy.array(file[f"facet{f+1}"]["surface_normals"]), device=device
                )
                for i, point_data in enumerate(points_data):
                    surface_points_with_facets[f, i, :] = point_data
                for i, normal_data in enumerate(normals_data):
                    surface_normals_with_facets[f, i, :] = normal_data

        log.info("Loading PAINT data complete")

        # Select only selected number of points to reduce compute.
        surface_points_with_facets = surface_points_with_facets[:, :: self.step_size]
        surface_normals_with_facets = surface_normals_with_facets[:, :: self.step_size]

        # Convert to 4D format.
        facet_translation_vectors = utils.convert_3d_direction_to_4d_format(
            facet_translation_vectors, device=device
        )

        # If we are learning the surface points from ``PAINT``, we do not need to translate the facets.
        if conversion_method == config_dictionary.convert_nurbs_from_points:
            facet_translation_vectors = torch.zeros(
                facet_translation_vectors.shape, device=device
            )

        # Convert to 4D format.
        canting_n = utils.convert_3d_direction_to_4d_format(canting_n, device=device)
        canting_e = utils.convert_3d_direction_to_4d_format(canting_e, device=device)
        surface_points_with_facets = utils.convert_3d_points_to_4d_format(
            surface_points_with_facets, device=device
        )
        surface_normals_with_facets = utils.convert_3d_direction_to_4d_format(
            surface_normals_with_facets, device=device
        )

        # Convert to NURBS surface.
        log.info("Converting to NURBS surface")
        facet_config_list = []
        for i in range(number_of_facets):
            log.info(f"Converting facet {i+1} of {number_of_facets}.")
            nurbs_surface = self.fit_nurbs_surface(
                surface_points=surface_points_with_facets[i],
                surface_normals=surface_normals_with_facets[i],
                conversion_method=conversion_method,
                number_control_points_e=number_control_points_e,
                number_control_points_n=number_control_points_n,
                degree_e=degree_e,
                degree_n=degree_n,
                tolerance=tolerance,
                initial_learning_rate=initial_learning_rate,
                max_epoch=max_epoch,
                device=device,
            )
            facet_config_list.append(
                FacetConfig(
                    facet_key=f"facet{i+1}",
                    control_points=nurbs_surface.control_points.detach(),
                    degree_e=nurbs_surface.degree_e,
                    degree_n=nurbs_surface.degree_n,
                    number_eval_points_e=number_eval_points_e,
                    number_eval_points_n=number_eval_points_n,
                    width=width,
                    height=height,
                    translation_vector=facet_translation_vectors[i],
                    canting_e=canting_e[i],
                    canting_n=canting_n[i],
                )
            )
        log.info("Surface configuration based on STRAL data complete!")
        return facet_config_list
