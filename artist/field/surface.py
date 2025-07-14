import torch

from artist.scenario.configuration_classes import SurfaceConfig
from artist.util import utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces


class Surface:
    """
    Implement the surface module which contains a list of facets.

    Attributes
    ----------
    nurbs_facets : List[NURBSSurface]
        A list of one nurbs surface for each facet.
    facet_translation_vectors : torch.Tensor
        The facet translation vectors for all facets.

    Methods
    -------
    get_surface_points_and_normals()
        Calculate all surface points and normals from all facets.
    """

    def __init__(
        self, surface_config: SurfaceConfig, device: torch.device | None = None
    ) -> None:
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
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        number_of_facets = len(surface_config.facet_list)

        self.nurbs_facets = []
        self.facet_translation_vectors = torch.empty(
            (number_of_facets, 4), device=device
        )
        for facet_index, facet_config in enumerate(surface_config.facet_list):
            self.nurbs_facets.append(
                NURBSSurfaces(
                    degrees=facet_config.degrees.unsqueeze(0).unsqueeze(0).expand(1, number_of_facets, -1),
                    control_points=facet_config.control_points.unsqueeze(0).unsqueeze(0).expand(1, number_of_facets, -1, -1, -1),
                    device=device,
                )
            )
            self.facet_translation_vectors[facet_index] = (
                facet_config.translation_vector
            )

    def get_surface_points_and_normals(
        self,
        number_of_points_per_facet: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Note: This method should only be used once nurbs are no longer being optimized.
        The returned surface points and normals are detached from the computational graph.

        Parameters
        ----------
        number_of_points_per_facet : torch.Tensor
            The number of surface points per facet.
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

        evaluation_points = utils.create_nurbs_evaluation_grid(
            number_of_evaluation_points=number_of_points_per_facet, device=device
        )

        surface_points = torch.empty(
            len(self.nurbs_facets), evaluation_points.shape[0], 4, device=device
        )
        surface_normals = torch.empty(
            len(self.nurbs_facets), evaluation_points.shape[0], 4, device=device
        )
        for i, nurbs_facet in enumerate(self.nurbs_facets):
            (
                facet_points,
                facet_normals,
            ) = nurbs_facet.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points, device=device
            )
            surface_points[i] = (
                facet_points + self.facet_translation_vectors[i]
            ).detach()
            surface_normals[i] = facet_normals.detach()
        return surface_points, surface_normals
