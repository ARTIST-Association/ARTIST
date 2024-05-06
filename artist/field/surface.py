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
