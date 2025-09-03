 @staticmethod
    def perform_inverse_canting_and_translation(
        canted_points: torch.Tensor,
        translation: torch.Tensor,
        canting: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Invert the canting rotation and translation on a batch of facets.

        Parameters
        ----------
        canted_points : torch.Tensor
            Homogeneous points after the forward transform, shape (number_of_facets, number_of_points, 4).
        translation : torch.Tensor
            Batch of facet translations, shape (number_of_facets, 4).
        canting : torch.Tensor
            Batch of canting vectors (east, north), shape (number_of_facets, 2, 4).
        device : torch.device | None
            Computation device.

        Returns
        -------
        torch.Tensor
            Original 3D points, shape (number_of_facets, number_of_points, 3).
        """
        device = get_device(device=device)
        number_of_facets, number_of_points, _ = canted_points.shape

        # Build forward transform per facet (use only ENU 3D for rotation).
        forward_transform = torch.zeros((number_of_facets, 4, 4), device=device)

        east_unit_vector = torch.nn.functional.normalize(
            canting[:, 0, :3], dim=1
        )  # (F, 3).
        north_unit_vector = torch.nn.functional.normalize(
            canting[:, 1, :3], dim=1
        )  # (F, 3).
        up_unit_vector = torch.nn.functional.normalize(
            torch.linalg.cross(east_unit_vector, north_unit_vector, dim=1), dim=1
        )  # (F, 3).

        forward_transform[:, :3, 0] = east_unit_vector
        forward_transform[:, :3, 1] = north_unit_vector
        forward_transform[:, :3, 2] = up_unit_vector
        # Translation column; ensure bottom element is 1.
        forward_transform[:, :3, 3] = translation[:, :3]
        forward_transform[:, 3, 3] = 1.0

        # Extract rotation and translation.
        rotation_matrix = forward_transform[:, :3, :3]  # (F, 3, 3).
        translation_vector = forward_transform[:, :3, 3]  # (F, 3).

        # Compute inverse transform.
        rotation_matrix_inverse = rotation_matrix.transpose(1, 2)  # (F, 3, 3).
        translation_inverse = -torch.bmm(
            rotation_matrix_inverse, translation_vector.unsqueeze(-1)
        ).squeeze(-1)  # (F, 3).

        inverse_transform = torch.zeros((number_of_facets, 4, 4), device=device)
        inverse_transform[:, :3, :3] = rotation_matrix_inverse
        inverse_transform[:, :3, 3] = translation_inverse
        inverse_transform[:, 3, 3] = 1.0

        # Apply inverse transform.
        restored_points = torch.bmm(canted_points, inverse_transform.transpose(1, 2))
        return restored_points[..., :3]



# Plot Settings.
helmholtz_colors = {
    "hgfblue": "#005AA0",
    "hgfdarkblue": "#0A2D6E",
    "hgfgreen": "#8CB423",
    "hgfgray": "#5A696E",
    "hgfaerospace": "#50C8AA",
    "hgfearthandenvironment": "#326469",
    "hgfenergy": "#FFD228",
    "hgfhealth": "#D23264",
    "hgfkeytechnologies": "#A0235A",
    "hgfmatter": "#F0781E",
}
