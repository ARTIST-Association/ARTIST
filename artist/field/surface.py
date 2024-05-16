from typing import Tuple

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

    Methods
    -------
    get_surface_points_and_normals()
        Calculate all surface points and normals from all facets.
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
                translation_vector=facet_config.translation_vector,
                canting_e=facet_config.canting_e,
                canting_n=facet_config.canting_n,
            )
            for facet_config in surface_config.facets_list
        ]

    def get_surface_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface normals.
        """
        eval_point_per_facet = (
            self.facets[0].number_eval_points_n * self.facets[0].number_eval_points_e
        )
        surface_points = torch.empty(len(self.facets), eval_point_per_facet, 4)
        surface_normals = torch.empty(len(self.facets), eval_point_per_facet, 4)
        for i, facet in enumerate(self.facets):
            facet_surface = facet.create_nurbs_surface()
            (
                facet_points,
                facet_normals,
            ) = facet_surface.calculate_surface_points_and_normals()
            surface_points[i] = facet_points + facet.translation_vector
            surface_normals[i] = facet_normals
        return surface_points, surface_normals
