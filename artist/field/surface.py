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
    nurbs_facets : list[NURBSSurface]
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
        Initialize the surface of one heliostat.

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
            If None, ''ARTIST'' will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.nurbs_facets = []

        for facet_config in surface_config.facet_list:
            self.nurbs_facets.append(
                NURBSSurfaces(
                    degrees=facet_config.degrees,
                    control_points=facet_config.control_points.unsqueeze(0).unsqueeze(
                        0
                    ),
                    device=device,
                )
            )

    def get_surface_points_and_normals(
        self,
        number_of_points_per_facet: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Parameters
        ----------
        number_of_points_per_facet : torch.Tensor
            The number of surface points per facet in east and then in north direction.
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points for one heliostat, tensor of shape [number_of_facets, number_of_surface_points_per_facet, 4].
        torch.Tensor
            The surface normals for one heliostat, tensor of shape [number_of_facets, number_of_surface_normals_per_facet, 4].
        """
        device = get_device(device=device)

        evaluation_points = utils.create_nurbs_evaluation_grid(
            number_of_evaluation_points=number_of_points_per_facet, device=device
        )

        # The surface points and surface normals will be returned as tensors of shape:
        # [number_of_facets, number_of_surface_points_per_facet, 4] and
        # [number_of_facets, number_of_surface_normals_per_facet, 4].
        surface_points = torch.empty(
            len(self.nurbs_facets), evaluation_points.shape[0], 4, device=device
        )
        surface_normals = torch.empty(
            len(self.nurbs_facets), evaluation_points.shape[0], 4, device=device
        )
        for i, nurbs_facet in enumerate(self.nurbs_facets):
            (
                surface_points[i],
                surface_normals[i],
            ) = nurbs_facet.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points.unsqueeze(0).unsqueeze(0),
                device=device,
            )
        return surface_points, surface_normals
