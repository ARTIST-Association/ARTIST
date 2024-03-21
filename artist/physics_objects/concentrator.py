import torch

from artist.physics_objects.module import AModule
from artist.util import artist_type_mapping_dict


class ConcentratorModule(AModule):
    """
    Implementation of the concentrator module.

    Attributes
    ----------
    facets : List[AFacetModule]
        The facets of the concentrator.

    See Also
    --------
    :class:AModule : Reference to the parent class.
    """

    def __init__(
        self,
        facets_type: str,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> None:
        """
        Initialize the concentrator.

        Parameters
        ----------
        facets_type : str
            The facet type, for example point cloud.
        surface_points : torch.Tensor
            The surface points on the concentrator.
        surface_normals : torch.Tensor
            The corresponding normal vectors to the points.
        """
        super().__init__()
        try:
            self.facets = artist_type_mapping_dict.facet_type_mapping[facets_type](
                surface_points=surface_points, surface_normals=surface_normals
            )
        except KeyError:
            raise KeyError(
                f"Currently the selected facet type: {facets_type} is not supported."
            )
