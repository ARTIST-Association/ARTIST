import logging

import torch

from artist.scenario.configuration_classes import FacetConfig, SurfaceConfig
from artist.util import config_dictionary, index_mapping, utils
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
        The number of NURBS control points along each direction of each 2D facet.
        Tensor of shape [2].
    degrees : torch.Tensor
        Degree of the NURBS along each direction of each 2D facet.
        Tensor of shape [2].

    Methods
    -------
    fit_nurbs()
        Fit a NURBS surface.
    generate_fitted_surface_config()
        Generate a fitted surface configuration.
    generate_ideal_surface_config()
        Generate an ideal surface configuration.
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
        exported to a binary file or downloaded from ``PAINT``. The data formats are different depending on their source.
        To convert this data into a surface configuration format suitable for ``ARTIST``, this converter first loads the
        data and then learns or creates NURBS surfaces based on the data. Finally, the converter returns a list of facets
        that can be used directly in an ``ARTIST`` scenario.

        Parameters
        ----------
        number_of_control_points : torch.Tensor
            The number of NURBS control points along each direction of each 2D facet (default is torch.tensor([10,10])).
            Tensor of shape [2].
        degrees : torch.Tensor
            Degree of the NURBS along each direction of each 2D facet (default is torch.tensor([3,3])).
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.number_of_control_points = number_of_control_points.to(device)
        self.degrees = degrees.to(device)

    def fit_nurbs(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        fit_method: str = config_dictionary.fit_nurbs_from_normals,
        tolerance: float = 1e-10,
        max_epoch: int = 400,
        device: torch.device | None = None,
    ) -> NURBSSurfaces:
        """
        Fit a NURBS surface.

        The surface points are first normalized and shifted to the range (0,1) to be compatible with the knot vector of
        the NURBS surface. The NURBS surface is then initialized with the correct number of control points, degrees, and
        knots. The origin of the control points is set based on the width and height of the point cloud. The control
        points are then fitted to the surface points or surface normals using the provided optimizer.

        Parameters
        ----------
        surface_points : torch.Tensor
            The surface points.
            Tensor of shape [number_of_surface_points, 4].
        surface_normals : torch.Tensor
            The surface normals.
            Tensor of shape [number_of_surface_points, 4].
        optimizer : torch.optim.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler.LRScheduler | None
            The learning rate scheduler (default is None).
        fit_method : str
            The method used to fit the NURBS, either from deflectometry points or normals (default is config_dictionary.fit_nurbs_from_normals).
        tolerance : float
            The tolerance value used for fitting NURBS surfaces (default is 1e-10).
        max_epoch : int
            The maximum number of epochs for the NURBS fit (default is 400).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            If the NURBS fit method is unknown.

        Returns
        -------
        NURBSSurfaces
            A fitted NURBS surface.
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
        evaluation_points[:, index_mapping.u] = 0

        # Initialize the NURBS surface.
        control_points = torch.zeros(
            (
                1,
                1,
                self.number_of_control_points[index_mapping.nurbs_u],
                self.number_of_control_points[index_mapping.nurbs_v],
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
            self.number_of_control_points[index_mapping.nurbs_u],
            device=device,
        )
        origin_offsets_n = torch.linspace(
            -height_of_nurbs / 2,
            height_of_nurbs / 2,
            self.number_of_control_points[index_mapping.nurbs_v],
            device=device,
        )

        control_points_e, control_points_n = torch.meshgrid(
            origin_offsets_e, origin_offsets_n, indexing="ij"
        )

        control_points[:, :, :, :, index_mapping.e] = control_points_e
        control_points[:, :, :, :, index_mapping.n] = control_points_n
        control_points[:, :, :, :, index_mapping.u] = 0

        # Since NURBS are only defined between (0,1), we need to normalize the evaluation points and remove the boundary points.
        evaluation_points[:, : index_mapping.u] = utils.normalize_points(
            evaluation_points[:, : index_mapping.u]
        )
        evaluation_points = evaluation_points.unsqueeze(0).unsqueeze(0)

        nurbs_surface = NURBSSurfaces(
            degrees=self.degrees,
            control_points=control_points,
            device=device,
        )

        # Add optimizable parameters (control points of the NURBS surface) to the optimizer.
        optimizer.param_groups.clear()
        optimizer.add_param_group(
            {"params": nurbs_surface.control_points.requires_grad_()}
        )

        loss = torch.inf
        epoch = 0
        while loss > tolerance and epoch <= max_epoch:
            points, normals = nurbs_surface.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points,
                canting=None,
                facet_translations=None,
                device=device,
            )

            optimizer.zero_grad()
            loss_function = torch.nn.MSELoss()

            if fit_method == config_dictionary.fit_nurbs_from_points:
                loss = loss_function(points, surface_points.unsqueeze(0).unsqueeze(0))
            else:
                loss = loss_function(normals, surface_normals.unsqueeze(0).unsqueeze(0))

            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step(loss.abs().mean().detach())
            if epoch % 100 == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}."
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        deflectometry_step_size: int = 100,
        fit_method: str = config_dictionary.fit_nurbs_from_normals,
        tolerance: float = 1e-10,
        max_epoch: int = 400,
        device: torch.device | None = None,
    ) -> SurfaceConfig:
        """
        Generate a fitted surface configuration.

        The fitted surface configuration is composed of separate facets. Each facet is defined by fitted control points,
        meaning the control points are fitted to measured point cloud or surface normals data. Initializing a surface
        from this configuration results in an imperfect heliostat surface with dents or bulges, reflecting real-world
        conditions. The surface can be fitted to deflectometry data or any other provided point cloud data.

        Parameters
        ----------
        heliostat_name : str
            The heliostat name, used for logging.
        facet_translation_vectors : torch.Tensor
            Translation vectors for each facet from heliostat origin to relative position.
            Tensor of shape [number_of_facets, 4].
        canting : torch.Tensor
            The canting vectors per facet in east and north directions
            Tensor of shape [number_of_facets, 2, 4].
        surface_points_with_facets_list : list[torch.Tensor]
            A list of facetted surface points. Points per facet may vary.
            Tensors in list of shape [number_of_points, 3].
        surface_normals_with_facets_list : list[torch.Tensor]
            A list of facetted surface normals. Points per facet may vary.
            Tensors in list of shape [number_of_points, 3].
        optimizer : torch.optim.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler.LRScheduler | None
            The learning rate scheduler (default is None).
        deflectometry_step_size : int
            The step size used to reduce the number of deflectometry points and normals for compute efficiency (default is 100).
        fit_method : str
            The method used to fit the NURBS, either from deflectometry points or normals (default is config_dictionary.fit_nurbs_from_normals).
        tolerance : float
            The tolerance value used for fitting NURBS surfaces (default is 1e-10).
        max_epoch : int
            The maximum number of epochs for the NURBS fit (default is 400).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        SurfaceConfig
            A surface configuration.
        """
        device = get_device(device=device)

        log.info("Beginning generation of the fitted surface configuration.")

        # All single_facet_surface_points and single_facet_surface_normals must have the same
        # dimensions, so that they can be stacked into a single tensor to be used by ARTIST.
        minimum_number_of_surface_points_all_facets = min(
            single_facet_surface_points.shape[
                index_mapping.number_of_points_or_normals_per_facet
            ]
            for single_facet_surface_points in surface_points_with_facets_list
        )
        reduced_single_facet_surface_points = [
            single_facet_surface_points[:minimum_number_of_surface_points_all_facets]
            for single_facet_surface_points in surface_points_with_facets_list
        ]
        surface_points_with_facets = torch.stack(reduced_single_facet_surface_points)

        minimum_number_of_surface_normals_all_facets = min(
            single_facet_surface_normals.shape[
                index_mapping.number_of_points_or_normals_per_facet
            ]
            for single_facet_surface_normals in surface_normals_with_facets_list
        )
        reduced_single_facet_surface_normals = [
            single_facet_surface_normals[:minimum_number_of_surface_normals_all_facets]
            for single_facet_surface_normals in surface_normals_with_facets_list
        ]
        surface_normals_with_facets = torch.stack(reduced_single_facet_surface_normals)

        # Select only a subset of points to reduce compute.
        surface_points_with_facets = surface_points_with_facets[
            :, ::deflectometry_step_size
        ]
        surface_normals_with_facets = surface_normals_with_facets[
            :, ::deflectometry_step_size
        ]

        # If a point cloud is used to learn the points, the facets translation is automatically learned.
        if fit_method == config_dictionary.fit_nurbs_from_points:
            facet_translation_vectors = torch.zeros(
                facet_translation_vectors.shape, device=device
            )

        # Convert to 4D format.
        surface_points_with_facets = utils.convert_3d_points_to_4d_format(
            surface_points_with_facets, device=device
        )
        surface_normals_with_facets = utils.convert_3d_directions_to_4d_format(
            surface_normals_with_facets, device=device
        )

        # Generate NURBS surface from multiple facets.
        # Each facet automatically has the same control point dimensions. This is required in ``ARTIST``.
        log.info(f"Generating NURBS surface for heliostat: {heliostat_name}.")
        facet_config_list = []
        for i in range(
            surface_points_with_facets.shape[index_mapping.number_of_facets]
        ):
            log.info(
                f"Generating facet {i + 1} of {surface_points_with_facets.shape[index_mapping.number_of_facets]}"
                "."
            )
            nurbs = self.fit_nurbs(
                surface_points=surface_points_with_facets[i],
                surface_normals=surface_normals_with_facets[i],
                optimizer=optimizer,
                scheduler=scheduler,
                fit_method=fit_method,
                tolerance=tolerance,
                max_epoch=max_epoch,
                device=device,
            )

            # During the NURBS fit, the control points were updated to represent real-world surfaces, they implicitly
            # learned the canting, but each facet is still centered around the origin, therefore a translation for each
            # facet is necessary.
            translated_control_points = (
                nurbs.control_points[0, 0] + facet_translation_vectors[i, :3]
            )

            facet_config_list.append(
                FacetConfig(
                    facet_key=f"facet_{i + 1}",
                    control_points=translated_control_points.detach(),
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

        The ideal surface configuration is composed of separate facets. Each facet is defined by ideal control points,
        meaning the control points start as 3D points on a flat, equidistant grid around the origin. These control points
        are then canted (rotated) and translated to the facet positions. Initializing a surface from this configuration
        results in an ideal heliostat surface without dents or bulges but with canting. This ideal heliostat surface can
        be used as a starting point for a surface reconstruction based on measured flux distributions.

        Parameters
        ----------
        facet_translation_vectors : torch.Tensor
            Translation vector for each facet from heliostat origin to relative position.
            Tensor of shape [number_of_facets, 4].
        canting : torch.Tensor
            The canting vector per facet in east and north direction.
            Tensor of shape [number_of_facets, 2, 4].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        SurfaceConfig
            A surface configuration.
        """
        device = get_device(device=device)

        log.info("Beginning generation of the ideal surface configuration.")
        facet_config_list = []

        control_points = utils.create_ideal_canted_nurbs_control_points(
            number_of_control_points=self.number_of_control_points,
            canting=canting,
            facet_translation_vectors=facet_translation_vectors,
            device=device,
        )

        for facet_index in range(facet_translation_vectors.shape[0]):
            facet_config = FacetConfig(
                facet_key=f"facet_{facet_index + 1}",
                control_points=control_points[facet_index],
                degrees=self.degrees,
                translation_vector=facet_translation_vectors[facet_index],
                canting=canting[facet_index],
            )
            facet_config_list.append(facet_config)

        surface_config = SurfaceConfig(facet_list=facet_config_list)

        log.info("Surface configuration based on ideal heliostat complete!")

        return surface_config
