import torch

from artist.field.facets_nurbs import NurbsFacet
from artist.util.configuration_classes import SurfaceConfig


class Surface(torch.nn.Module):
    """
    Implementation of the surface module which contains a list of facets.

    Attributes
    ----------
    facets : List[Facet]
        A list of facets that comprise the surface of the heliostat.
    """

    def __init__(self, surface_config: SurfaceConfig) -> None:
        """
        Initialize the surface.

        Parameters
        ----------
        surface_config : SurfaceConfig
            The surface configuration parameters used to construct the surface.
        """
        super(Surface, self).__init__()
        self.facets = [
            NurbsFacet(
                control_points=facet_config.control_points,
                degree_e=facet_config.degree_e,
                degree_n=facet_config.degree_n,
                number_eval_points_e=facet_config.number_eval_points_e,
                number_eval_points_n=facet_config.number_eval_points_n,
                width=facet_config.width,
                height=facet_config.height,
                position=facet_config.position,
                canting_e=facet_config.canting_e,
                canting_n=facet_config.canting_n,
            )
            for facet_config in surface_config.facets_list
        ]

    def get_surface_points_and_normals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface_normals.
        """
        surface_points = []
        surface_normals = []
        for facet in self.facets:
            facet_surface = facet.create_nurbs_surface()
            (
                facet_points,
                facet_normals,
            ) = facet_surface.calculate_surface_points_and_normals()
            surface_points.append(facet_points + facet.position)
            surface_normals.append(facet_normals)
        surface_points = torch.stack(surface_points, dim=1).permute(1, 0, 2)
        surface_normals = torch.stack(surface_normals, dim=1).permute(1, 0, 2)
        return surface_points, surface_normals
