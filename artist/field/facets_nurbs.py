import torch

from artist.field.nurbs import NURBSSurface


class NurbsFacet(torch.nn.Module):
    """
    Facet modeled with a NURBS surface.

    Attributes
    ----------
    control_points_e : torch.Tensor
        The control points in the east direction of the NURBS surface.
    control_points_u : torch.Tensor
        The control points in the up direction of the NURBS surface.
    knots_e : torch.Tensor
        The knots in the east direction for the NURBS surface.
    knots_u : torch.Tensor
        The knots in the up direction for the NURBS surface.
    width : float
        The width of the facet.
    height : float
        The height of the facet.
    position : torch.Tensor
        The position of the facet.
    canting_e : torch.Tensor
        The canting vector in the east direction of the facet.
    canting_u : torch.Tensor
        The canting vector in the up direction of the facet.
    """

    def __init__(
        self,
        control_points_e: torch.Tensor,
        control_points_u: torch.Tensor,
        knots_e: torch.Tensor,
        knots_u: torch.Tensor,
        width: float,
        height: float,
        position: torch.Tensor,
        canting_e: torch.Tensor,
        canting_u: torch.Tensor,
    ) -> None:
        """
        Initialize a nurbs facet.

        Parameters
        ----------
        control_points_e : torch.Tensor
            The control points in the east direction of the NURBS surface.
        control_points_u : torch.Tensor
            The control points in the up direction of the NURBS surface.
        knots_e : torch.Tensor
            The knots in the east direction for the NURBS surface.
        knots_u : torch.Tensor
            The knots in the up direction for the NURBS surface.
        width : float
            The width of the facet.
        height : float
            The height of the facet.
        position : torch.Tensor
            The position of the facet.
        canting_e : torch.Tensor
            The canting vector in the east direction of the facet.
        canting_u : torch.Tensor
            The canting vector in the up direction of the facet.
        """
        super(NurbsFacet, self).__init__()
        self.control_points_e = control_points_e
        self.control_points_u = control_points_u
        self.knots_e = knots_e
        self.knots_u = knots_u
        self.width = width
        self.height = height
        self.position = position
        self.canting_e = canting_e
        self.canting_u = canting_u

    def create_nurbs_surface(self) -> NURBSSurface:
        """
        Create a NURBS surface to model a facet.

        Returns
        -------
        NURBSSurface
            The NURBS surface of one facet.
        """
        evaluation_points_e = torch.linspace(0, 1, self.number_eval_points_e)
        evaluation_points_n = torch.linspace(0, 1, self.number_eval_points_n)

        nurbs_surface = NURBSSurface(
            self.degree_e,
            self.degree_n,
            evaluation_points_e,
            evaluation_points_n,
            self.control_points,
        )
        return nurbs_surface
