import torch

from artist.field.nurbs import NURBSSurface


class NurbsFacet(torch.nn.Module):
    """
    Facet modeled with a NURBS surface.

    Attributes
    ----------
    control_points : torch.Tensor
        The control points of the NURBS surface.
    degree_e : int
        The degree of the NURBS surface in the east direction.
    degree_n : int
        The degree of the NURBS surface in the north direction.
    number_eval_points_e : int
        The number of evaluation points for the NUBRS surface in the east direction.
    number_eval_points_n : int
        The number of evaluation points for the NUBRS surface in the north direction.
    width : float
        The width of the facet.
    height : float
        The height of the facet.
    position : torch.Tensor
        The position of the facet.
    canting_e : torch.Tensor
        The canting vector in the east direction of the facet.
    canting_n : torch.Tensor
        The canting vector in the north direction of the facet.
    """

    def __init__(
        self,
        control_points: torch.Tensor,
        degree_e: int,
        degree_n: int,
        number_eval_points_e: int,
        number_eval_points_n: int,
        width: float,
        height: float,
        position: torch.Tensor,
        canting_e: torch.Tensor,
        canting_n: torch.Tensor,
    ) -> None:
        """
        Initialize a nurbs facet.

        Parameters
        ----------
        control_points : torch.Tensor
            The control points of the NURBS surface.
        degree_e : int
            The degree of the NURBS surface in the east direction.
        degree_n : int
            The degree of the NURBS surface in the north direction.
        number_eval_points_e : int
            The number of evaluation points for the NUBRS surface in the east direction.
        number_eval_points_n : int
            The number of evaluation points for the NUBRS surface in the north direction.
        width : float
            The width of the facet.
        height : float
            The height of the facet.
        position : torch.Tensor
            The position of the facet.
        canting_e : torch.Tensor
            The canting vector in the east direction of the facet.
        canting_n : torch.Tensor
            The canting vector in the north direction of the facet.
        """
        super(NurbsFacet, self).__init__()
        self.control_points = control_points
        self.degree_e = degree_e
        self.degree_n = degree_n
        self.number_eval_points_e = number_eval_points_e
        self.number_eval_points_n = number_eval_points_n
        self.width = width
        self.height = height
        self.position = position
        self.canting_e = canting_e
        self.canting_n = canting_n

    def create_nurbs_surface(self) -> NURBSSurface:
        """
        Create a NURBS surface to model a facet.

        Returns
        -------
        NURBSSurface
            The NURBS surface of one facet.
        """
        evaluation_points_e = torch.linspace(0 + 1e-5, 1 - 1e-5, self.number_eval_points_e)
        evaluation_points_n = torch.linspace(0 + 1e-5, 1 - 1e-5, self.number_eval_points_n)

        nurbs_surface = NURBSSurface(
            self.degree_e,
            self.degree_n,
            evaluation_points_e,
            evaluation_points_n,
            self.control_points,
        )
        return nurbs_surface
