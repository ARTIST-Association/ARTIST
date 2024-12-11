from typing import Union

import torch

from artist.util.nurbs import NURBSSurface


class NurbsFacet(torch.nn.Module):
    """
    Model a facet with a NURBS surface.

    Attributes
    ----------
    control_points : torch.Tensor
        The control points of the NURBS surface.
    degree_e : int
        The degree of the NURBS surface in the east direction.
    degree_n : int
        The degree of the NURBS surface in the north direction.
    number_eval_points_e : int
        The number of evaluation points for the NURBS surface in the east direction.
    number_eval_points_n : int
        The number of evaluation points for the NURBS surface in the north direction.
    translation_vector : torch.Tensor
        The translation_vector of the facet.
    canting_e : torch.Tensor
        The canting vector in the east direction of the facet.
    canting_n : torch.Tensor
        The canting vector in the north direction of the facet.


    Methods
    -------
    create_nurbs_surface()
        Create a NURBS surface to model a facet.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        control_points: torch.Tensor,
        degree_e: int,
        degree_n: int,
        number_eval_points_e: int,
        number_eval_points_n: int,
        translation_vector: torch.Tensor,
        canting_e: torch.Tensor,
        canting_n: torch.Tensor,
    ) -> None:
        """
        Initialize a NURBS facet.

        The heliostat surface can be divided into facets. In ARTIST, the surfaces are modeled using
        Non-Uniform Rational B-Splines (NURBS). Thus, each facet is an individual NURBS surface. The
        NURBS surface is created by specifying several parameters. For a detailed description of these
        parameters see the `NURBS-tutorial`. For now, note that the NURBS surfaces can be formed through
        control points, two degrees, the number of evaluation points in east and north direction, a
        translation vector to match the facets to their position, and canting vectors.

        Parameters
        ----------
        control_points : torch.Tensor
            The control points of the NURBS surface.
        degree_e : int
            The degree of the NURBS surface in the east direction.
        degree_n : int
            The degree of the NURBS surface in the north direction.
        number_eval_points_e : int
            The number of evaluation points for the NURBS surface in the east direction.
        number_eval_points_n : int
            The number of evaluation points for the NURBS surface in the north direction.
        translation_vector : torch.Tensor
            The translation_vector of the facet.
        canting_e : torch.Tensor
            The canting vector in the east direction of the facet.
        canting_n : torch.Tensor
            The canting vector in the north direction of the facet.
        """
        super().__init__()
        self.control_points = control_points
        self.degree_e = degree_e
        self.degree_n = degree_n
        self.number_eval_points_e = number_eval_points_e
        self.number_eval_points_n = number_eval_points_n
        self.translation_vector = translation_vector
        self.canting_e = canting_e
        self.canting_n = canting_n

    def create_nurbs_surface(
        self, device: Union[torch.device, str] = "cuda"
    ) -> NURBSSurface:
        """
        Create a NURBS surface to model a facet.

        Parameters
        ----------
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        NURBSSurface
            The NURBS surface of one facet.
        """
        device = torch.device(device)
        # Since NURBS are only defined between (0,1), a small offset is required to exclude the boundaries from the
        # defined evaluation points.
        evaluation_points_rows = torch.linspace(
            0 + 1e-5, 1 - 1e-5, self.number_eval_points_e, device=device
        )
        evaluation_points_cols = torch.linspace(
            0 + 1e-5, 1 - 1e-5, self.number_eval_points_n, device=device
        )
        evaluation_points = torch.cartesian_prod(
            evaluation_points_rows, evaluation_points_cols
        )

        evaluation_points = evaluation_points.to(device)

        evaluation_points_e = evaluation_points[:, 0]
        evaluation_points_n = evaluation_points[:, 1]

        nurbs_surface = NURBSSurface(
            self.degree_e,
            self.degree_n,
            evaluation_points_e,
            evaluation_points_n,
            self.control_points,
            device,
        )
        return nurbs_surface

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
