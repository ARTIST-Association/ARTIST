from typing import Union, Optional

import torch

from artist.util.nurbs import NURBSSurface


class NurbsFacet(torch.nn.Module):
    """
    Model a facet with a NURBS surface.

    Attributes
    ----------
    control_points_ideal : torch.Tensor
        The control points of the ideal NURBS surface derived from e.g. construction files like CAD. All z values are 0 if not given.
    control_points_measured : torch.Tensor
        The control points of the NURBS surface given as the difference to the ideal.
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
        degree_e: int,
        degree_n: int,
        number_eval_points_e: int,
        number_eval_points_n: int,
        translation_vector: torch.Tensor,
        control_points: torch.Tensor,
    ) -> None:
        super().__init__()

        self.control_points = control_points

        self.degree_e = degree_e
        self.degree_n = degree_n
        self.number_eval_points_e = number_eval_points_e
        self.number_eval_points_n = number_eval_points_n
        self.translation_vector = translation_vector

    def create_nurbs_surface(
        self,
        control_point_type: Optional[str] = None, 
        device: Union[torch.device, str] = "cuda"

    ) -> NURBSSurface:
        """
        Create a NURBS surface to model a facet.

        Parameters
        ----------
        control_point_type : Optional[str]
            The type of control points to use.
            String options are:
            - "ideal": Use the ideal/prototyped control points.
            - "measured": Use the measured/derived control points.
            - None: Use the combined control points.
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

        if control_point_type == "ideal":
            control_points = self.control_points_ideal
        elif control_point_type == "measured":
            control_points = self.control_points_measured
        else:
            control_points = self.control_points

        nurbs_surface = NURBSSurface(
            degree_e = self.degree_e,
            degree_n = self.degree_n,
            evaluation_points_e = evaluation_points_e,
            evaluation_points_n = evaluation_points_n,
            control_points= control_points,
            device = device,
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
