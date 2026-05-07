import torch

from artist.nurbs.surfaces import NURBSSurfaces
from artist.nurbs.utils import create_nurbs_evaluation_grid
from artist.util.config import SurfaceConfig
from artist.util import indices
from artist.util.env import get_device


class Surface:
    """
    Implement the surface module from a list of facets.

    Attributes
    ----------
    nurbs_surface : NURBSSurface
        The NURBS surface.

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
        on the heliostat, not the whole heliostat. The surface can be aligned through the kinematics and
        its actuators. Each surface and thus each facet is defined through NURBS, the discrete surface
        points and surface normals can be retrieved.

        Parameters
        ----------
        surface_config : SurfaceConfig
            The surface configuration parameters used to construct the surface.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        degrees = surface_config.facet_list[indices.first_facet].degrees
        control_points = []

        for facet_config in surface_config.facet_list:
            control_points.append(facet_config.control_points)

        control_points = torch.stack(control_points)

        self.nurbs_surface = NURBSSurfaces(
            degrees=degrees,
            control_points=control_points.unsqueeze(indices.heliostat_dimension),
            device=device,
        )

    def get_surface_points_and_normals(
        self,
        number_of_points_per_facet: torch.Tensor,
        canting: torch.Tensor,
        facet_translations: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Parameters
        ----------
        number_of_points_per_facet : torch.Tensor
            The number of sampling points along each direction of each 2D facet.
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points for one heliostat, tensor of shape [number_of_facets, number_of_surface_points_per_facet, 4].
        torch.Tensor
            The surface normals for one heliostat, tensor of shape [number_of_facets, number_of_surface_normals_per_facet, 4].
        """
        device = get_device(device=device)

        evaluation_points = (
            create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_points_per_facet, device=device
            )
            .unsqueeze(indices.heliostat_dimension)
            .unsqueeze(indices.facet_index_unbatched)
            .expand(1, self.nurbs_surface.number_of_facets_per_surface, -1, -1)
        )

        if torch.all(self.nurbs_surface.control_points[..., 2] == 0):
            (
                surface_points,
                surface_normals,
            ) = self.nurbs_surface.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points,
                canting=canting.unsqueeze(indices.heliostat_dimension),
                facet_translations=facet_translations.unsqueeze(
                    indices.heliostat_dimension
                ),
                device=device,
            )
        else:
            (
                surface_points,
                surface_normals,
            ) = self.nurbs_surface.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points,
                canting=None,
                facet_translations=None,
                device=device,
            )
        return surface_points, surface_normals
