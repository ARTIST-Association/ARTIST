import logging

import torch

from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces

log = logging.getLogger(__name__)
"""A logger for the surface generator."""


class SurfaceGenerator:
    """
    A surface generator for fitted and ideal surfaces.

    Attributes
    ----------
    number_of_control_points : torch.Tensor
        Number of NURBS control points per facet in the east an north direction.
    degrees : torch.Tensor
        Degree of the NURBS in the east and north direction.

    Methods
    -------
    fit_nurbs_surface()
        Fit the NURBS surface given the conversion method.
    generate_fitted_surface_config()
        Generate a fitted surface configuration.
    generate_ideal_surface_config()
        Generate an ideal surface configuration.
    perform_canting_and_translation()
        Perform the canting rotation and facet translation.
    """

    def __init__(
        self,
        number_of_control_points: torch.Tensor = torch.tensor([10, 10]),
        degrees: torch.Tensor = torch.tensor([3, 3]),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface generator.

        Heliostat data, including information regarding their surfaces and structure, can be generated via ``STRAL`` and
        exported to a binary file or downloaded from ```PAINT``. The data formats are different depending on their source.
        To convert this data into a surface configuration format suitable for ``ARTIST``, this converter first loads the
        data and then learns NURBS surfaces based on the data. Finally, the converter returns a list of facets that can
        be used directly in an ``ARTIST`` scenario.

        Parameters
        ----------
        number_of_control_points : torch.Tensor
            Number of NURBS control points per facet in the east an north direction (default is torch.tensor([20, 20])).
        degrees : torch.Tensor
            Degree of the NURBS in the east and north direction (default is torch.tensor([3, 3])).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.number_of_control_points = number_of_control_points.to(device)
        self.degrees = degrees.to(device)

    def fit_nurbs(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        fit_method: str = config_dictionary.fit_nurbs_from_normals,
        tolerance: float = 3e-5,
        initial_learning_rate: float = 1e-3,
        max_epoch: int = 400,
        device: torch.device | None = None,
    ) -> NURBSSurfaces:
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
        fit_method : str
            The method used to fit the NURBS, either from deflectometry points or normals (default is config_dictionary.fit_nurbs_from_normals).
        tolerance : float
            The tolerance value used for fitting NURBS surfaces to deflectometry (default is 3e-5).
        initial_learning_rate : float
            The initial learning rate for the NURBS fit (default is 1e-3).
        max_epoch : int
            The maximum number of epochs for the NURBS fit (default is 400).                        
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            If the NURBS fit method is unknown.
            
        Returns
        -------
        NURBSSurface
            A NURBS surface.
        """
        accepted_conversion_methods = [
            config_dictionary.fit_nurbs_from_points,
            config_dictionary.fit_nurbs_from_normals,
        ]
        if fit_method not in accepted_conversion_methods:
            raise NotImplementedError(
                f"The conversion method '{fit_method}' is not yet supported in ARTIST."
            )

        device = get_device(device=device)

        evaluation_points = surface_points.clone()
        evaluation_points[:, 2] = 0

        # Initialize the NURBS surface.
        control_points = torch.zeros(
            (
                1,
                1,
                self.number_of_control_points[0],
                self.number_of_control_points[1],
                3,
            ),
            device=device,
        )

        width_of_nurbs = torch.max(evaluation_points[:, 0]) - torch.min(
            evaluation_points[:, 0]
        )
        height_of_nurbs = torch.max(evaluation_points[:, 1]) - torch.min(
            evaluation_points[:, 1]
        )

        origin_offsets_e = torch.linspace(
            -width_of_nurbs / 2,
            width_of_nurbs / 2,
            self.number_of_control_points[0],
            device=device,
        )
        origin_offsets_n = torch.linspace(
            -height_of_nurbs / 2,
            height_of_nurbs / 2,
            self.number_of_control_points[1],
            device=device,
        )

        control_points_e, control_points_n = torch.meshgrid(
            origin_offsets_e, origin_offsets_n, indexing="ij"
        )

        control_points[:, :, :, :, 0] = control_points_e
        control_points[:, :, :, :, 1] = control_points_n
        control_points[:, :, :, :, 2] = 0

        # Since NURBS are only defined between (0,1), we need to normalize the evaluation points and remove the boundary points.
        evaluation_points[:, 0] = utils.normalize_points(evaluation_points[:, 0])
        evaluation_points[:, 1] = utils.normalize_points(evaluation_points[:, 1])
        evaluation_points = evaluation_points.unsqueeze(0).unsqueeze(0)

        nurbs_surface = NURBSSurfaces(
            degrees=self.degrees,
            control_points=control_points,
            device=device,
        )

        # Optimize the control points of the NURBS surface.
        optimizer = torch.optim.Adam(
            [nurbs_surface.control_points.requires_grad_()],
            lr=initial_learning_rate,
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
        while loss > tolerance and epoch <= max_epoch:
            points, normals = nurbs_surface.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points, device=device
            )

            optimizer.zero_grad()

            loss_function = torch.nn.MSELoss()

            if fit_method == config_dictionary.fit_nurbs_from_points:
                loss = loss_function(
                    points,
                    surface_points.unsqueeze(0).unsqueeze(0)
                )
                #TODO
                #loss = (points - surface_points).abs().mean()
            else:
                loss = loss_function(
                    normals,
                    surface_normals.unsqueeze(0).unsqueeze(0)
                )
                #loss = (normals - surface_normals).abs().mean()

            loss.backward()

            optimizer.step()
            scheduler.step(loss.abs().mean())
            if epoch % 100 == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}.",
                )
            epoch += 1

        return nurbs_surface

    def generate_fitted_surface_config(
        self,
        heliostat_name: str,
        facet_translation_vectors: torch.Tensor,
        canting: torch.Tensor,
        surface_points_with_facets_list: list[torch.Tensor],
        surface_normals_with_facets_list: list[torch.Tensor],
        deflectometry_step_size: int = 100,
        fit_method: str = config_dictionary.fit_nurbs_from_normals,
        tolerance: float = 3e-5,
        initial_learning_rate: float = 1e-3,
        max_epoch: int = 400,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a fitted surface configuration.

        Parameters
        ----------
        heliostat_name : str
            The heliostat name, used for logging.
        facet_translation_vectors : torch.Tensor
            Translation vector for each facet from heliostat origin to relative position.
        canting : torch.Tensor
            The canting vector per facet in east and north direction.
        surface_points_with_facets_list : list[torch.Tensor]
            A list of facetted surface points. Points per facet may vary.
        surface_normals_with_facets_list : list[torch.Tensor]
            A list of facetted surface normals. Normals per facet may vary.
        deflectometry_step_size : torch.Tensor
            The step size used to reduce the number of deflectometry points and normals for compute efficiency (default is 100).
        fit_method : str
            The method used to fit the NURBS, either from deflectometry points or normals (default is config_dictionary.fit_nurbs_from_normals).
        tolerance : float
            The tolerance value used for fitting NURBS surfaces to deflectometry (default is 3e-5).
        initial_learning_rate : float
            The initial learning rate for the NURBS fit (default is 1e-3).
        max_epoch : int
            The maximum number of epochs for the NURBS fit (default is 400).
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

        log.info("Beginning generation of the fitted surface configuration.")

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
        surface_points_with_facets = surface_points_with_facets[:, :: deflectometry_step_size]
        surface_normals_with_facets = surface_normals_with_facets[:, :: deflectometry_step_size]

        # Convert to 4D format.
        facet_translation_vectors = utils.convert_3d_directions_to_4d_format(
            facet_translation_vectors, device=device
        )
        # If we are using a point cloud to learn the points, we do not need to translate the facets.
        if fit_method == config_dictionary.fit_nurbs_from_points:
            facet_translation_vectors = torch.zeros(
                facet_translation_vectors.shape, device=device
            )
        # Convert to 4D format.
        canting = utils.convert_3d_directions_to_4d_format(canting, device=device)
        surface_points_with_facets = utils.convert_3d_points_to_4d_format(
            surface_points_with_facets, device=device
        )
        surface_normals_with_facets = utils.convert_3d_directions_to_4d_format(
            surface_normals_with_facets, device=device
        )

        # Generate NURBS surface from multiple facets.
        # Each facet automatically has the same control points dimensions. This is required in ARTIST.
        log.info(f"Generating NURBS surface for heliostat: {heliostat_name}.")
        facet_config_list = []
        for i in range(surface_points_with_facets.shape[0]):
            log.info(
                f"Generating facet {i + 1} of {surface_points_with_facets.shape[0]}."
            )
            nurbs = self.fit_nurbs(
                surface_points=surface_points_with_facets[i],
                surface_normals=surface_normals_with_facets[i],
                fit_method=fit_method,
                tolerance=tolerance,
                initial_learning_rate=initial_learning_rate,
                max_epoch=max_epoch,
                device=device,
            )

            # Only a translation is necessary, the canting is learned, therefore the cantings are unit vectors.
            canted_control_points = self.perform_canting_and_translation(
                points=nurbs.control_points[0, 0].detach(),
                translation=facet_translation_vectors[i],
                canting=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device
                ),
                device=device,
            )

            facet_config_list.append(
                FacetConfig(
                    facet_key=f"facet_{i + 1}",
                    control_points=canted_control_points,
                    degrees=nurbs.degrees,
                    translation_vector=facet_translation_vectors[i],
                    canting=canting[i],
                )
            )

        surface_config = SurfaceConfig(facet_list=facet_config_list)
        log.info("Surface configuration based on fit complete!")
        return surface_config

    def generate_ideal_surface_config(
        self,
        facet_translation_vectors: torch.Tensor,
        canting: torch.Tensor,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate an ideal surface configuration.

        Parameters
        ----------
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

        log.info("Beginning generation of the ideal surface configuration.")
        facet_config_list = []

        # Convert to 4D format.
        facet_translation_vectors = utils.convert_3d_directions_to_4d_format(
            facet_translation_vectors, device=device
        )
        canting = utils.convert_3d_directions_to_4d_format(canting, device=device)

        for facet_index in range(facet_translation_vectors.shape[0]):
            control_points = torch.zeros(
                (self.number_of_control_points[0], self.number_of_control_points[1], 3),
                device=device,
            )
            origin_offsets_e = torch.linspace(
                -torch.norm(canting[facet_index], dim=-1)[0],
                torch.norm(canting[facet_index], dim=-1)[0],
                control_points.shape[0],
                device=device,
            )
            origin_offsets_n = torch.linspace(
                -torch.norm(canting[facet_index], dim=-1)[1],
                torch.norm(canting[facet_index], dim=-1)[1],
                control_points.shape[1],
                device=device,
            )

            control_points_e, control_points_n = torch.meshgrid(
                origin_offsets_e, origin_offsets_n, indexing="ij"
            )

            control_points[:, :, 0] = control_points_e
            control_points[:, :, 1] = control_points_n
            control_points[:, :, 2] = 0

            canted_control_points = self.perform_canting_and_translation(
                points=control_points,
                canting=canting[facet_index],
                translation=facet_translation_vectors[facet_index],
                device=device,
            )

            facet_config = FacetConfig(
                facet_key=f"facet_{facet_index + 1}",
                control_points=canted_control_points,
                degrees=self.degrees,
                translation_vector=facet_translation_vectors[facet_index],
                canting=canting[facet_index],
            )
            facet_config_list.append(facet_config)

        surface_config = SurfaceConfig(facet_list=facet_config_list)

        log.info("Surface configuration based on ideal heliostat complete!")

        return surface_config

    def perform_canting_and_translation(
        self,
        points: torch.Tensor,
        translation: torch.Tensor,
        canting: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Perform the canting rotation and facet translation.

        Parameters
        ----------
        points : torch.Tensor
            The points to be canted and translated.
        translation : torch.Tensor
            The facet translation vector.
        canting : torch.Tensor
            The canting vectors in east and north direction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The canted and translated points.
        """
        device = get_device(device=device)

        rotation_matrix = torch.zeros((4, 4), device=device)

        rotation_matrix[:, 0] = torch.nn.functional.normalize(canting[0], dim=0)
        rotation_matrix[:, 1] = torch.nn.functional.normalize(canting[1], dim=0)
        rotation_matrix[:3, 2] = torch.nn.functional.normalize(
            torch.linalg.cross(rotation_matrix[:3, 0], rotation_matrix[:3, 1]), dim=0
        )

        rotation_matrix[3, 3] = 1.0

        canted_points = (
            utils.convert_3d_points_to_4d_format(points=points, device=device).reshape(
                -1, 4
            )
            @ rotation_matrix.T
        ).reshape(points.shape[0], points.shape[1], 4)

        canted_with_translation = canted_points + translation

        return canted_with_translation[:, :, :3]
