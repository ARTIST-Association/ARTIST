from typing import List, Tuple

import torch


class AFacetModule(torch.nn.Module):
    """
    Abstract base class for all facet modules.

    See also
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
        return NotImplementedError("Please overwrite")

    def facetted_discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return NotImplementedError("Please overwrite")

    def get_facet_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Please overwrite")
