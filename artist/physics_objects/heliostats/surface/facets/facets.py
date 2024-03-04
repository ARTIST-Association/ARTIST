from typing import List, Tuple

import torch


class AFacetModule(torch.nn.Module):
    """
    Abstract base class for all facet modules.

    Attributes
    ----------
    positions : torch.Tensor
        [INSERT DESCRIPTION HERE!]
    spans_n : torch.Tensor
        [INSERT DESCRIPTION HERE!]
    spans_e : torch.Tensor
        [INSERT DESCRIPTION HERE!]
    _discrete_points : List[torch.Tensor]
        [INSERT DESCRIPTION HERE!]
    _discrete_points_ideal : List[torch.Tensor]
        [INSERT DESCRIPTION HERE!]
    _normals : List[torch.Tensor]
        [INSERT DESCRIPTION HERE!]
    _normals_ideal : List[torch.Tensor]
        [INSERT DESCRIPTION HERE!]
    offsets : torch.Tensor
        [INSERT DESCRIPTION HERE!]
    cant_rots : torch.Tensor
        [INSERT DESCRIPTION HERE!]

    Methods
    -------
    discrete_points_and_normals()
        [INSERT DESCRIPTION HERE!]
    faceted_discrete_points_and_normals()
        [INSERT DESCRIPTION HERE!]
    get_facet_surface()
        [INSERT DESCRIPTION HERE!]

    See Also
    --------
    :class: torch.nn.Module : Reference to the parent class.
    """

    # Relative to heliostat position.
    positions: torch.Tensor
    spans_n: torch.Tensor
    spans_e: torch.Tensor

    _discrete_points: List[torch.Tensor]
    _discrete_points_ideal: List[torch.Tensor]
    _normals: List[torch.Tensor]
    _normals_ideal: List[torch.Tensor]

    offsets: torch.Tensor

    cant_rots: torch.Tensor

    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [INSERT DESCRIPTION HERE!]
        """
        return NotImplementedError("Must be overwritten.")

    def faceted_discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [INSERT DESCRIPTION HERE!]
        """
        return NotImplementedError("Must be overwritten.")

    def get_facet_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [INSERT DESCRIPTION HERE!]
        """
        raise NotImplementedError("Must be overwritten.")
