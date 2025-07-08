import torch

from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurface


class NurbsFacet:
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
    """

    def __init__(
        self,
        control_points: torch.Tensor,
        degrees: torch.Tensor,
        number_eval_points: torch.Tensor,
        translation_vector: torch.Tensor,
        cantings: torch.Tensor
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
        self.control_points = control_points
        self.degrees = degrees
        self.number_eval_points = number_eval_points
        self.translation_vector = translation_vector
        self.cantings = cantings

    def create_nurbs_surface(self, device: torch.device | None = None) -> NURBSSurface:
        """
        Create a NURBS surface to model a facet.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        NURBSSurface
            The NURBS surface of one facet.
        """
        device = get_device(device=device)

        # Since NURBS are only defined between (0,1), a small offset is required to exclude the boundaries from the
        # defined evaluation points.
        evaluation_points_rows = torch.linspace(
            0 + 1e-5, 1 - 1e-5, self.number_eval_points[0], device=device
        )
        evaluation_points_cols = torch.linspace(
            0 + 1e-5, 1 - 1e-5, self.number_eval_points[1], device=device
        )
        evaluation_points = torch.cartesian_prod(
            evaluation_points_rows, evaluation_points_cols
        )

        nurbs_surface = NURBSSurface(
            degrees=self.degrees,
            evaluation_points=evaluation_points,
            control_points=self.control_points,
            device=device,
        )
        return nurbs_surface
