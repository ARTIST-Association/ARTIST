import torch

from artist.field.facets_nurbs import NurbsFacet
from artist.scenario.configuration_classes import SurfaceConfig
from artist.util.environment_setup import get_device


class Surface:
    """
    Implement the surface module which contains a list of facets.

    Attributes
    ----------
    facets : List[Facet]
        A list of facets that comprise the surface of a heliostat.

    Methods
    -------
    get_surface_points_and_normals()
        Calculate all surface points and normals from all facets.
    """

    def __init__(self, surface_config: SurfaceConfig) -> None:
        """
        Initialize the surface.

        The heliostat surface consists of one or more facets. The surface only describes the mirrors
        on the heliostat, not the whole heliostat. The surface can be aligned through the kinematic and
        its actuators. Each surface and thus each facet is defined through NURBS, the discrete surface
        points and surface normals can be retrieved.

        Parameters
        ----------
        surface_config : SurfaceConfig
            The surface configuration parameters used to construct the surface.
        """
        self.facets = [
            NurbsFacet(
                control_points=facet_config.control_points,
                degree_e=facet_config.degree_e,
                degree_n=facet_config.degree_n,
                number_eval_points_e=facet_config.number_eval_points_e,
                number_eval_points_n=facet_config.number_eval_points_n,
                translation_vector=facet_config.translation_vector,
                canting_e=facet_config.canting_e,
                canting_n=facet_config.canting_n,
            )
            for facet_config in surface_config.facet_list
        ]

    def get_surface_points_and_normals(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Note: This method should only be used once nurbs are no longer being optimized.
        The returned surface points and normals are detached from the computational graph.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points, detached from the computational graph.
        torch.Tensor
            The surface normals, detached from the computational graph.
        """
        device = get_device(device=device)

        eval_point_per_facet = (
            self.facets[0].number_eval_points_n * self.facets[0].number_eval_points_e
        )
        surface_points = torch.empty(
            len(self.facets), eval_point_per_facet, 4, device=device
        )
        surface_normals = torch.empty(
            len(self.facets), eval_point_per_facet, 4, device=device
        )
        for i, facet in enumerate(self.facets):
            facet_surface = facet.create_nurbs_surface(device=device)
            (
                facet_points,
                facet_normals,
            ) = facet_surface.calculate_surface_points_and_normals(device=device)
            surface_points[i] = (facet_points + facet.translation_vector).detach()
            surface_normals[i] = facet_normals.detach()
        return surface_points, surface_normals
