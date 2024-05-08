class Rays:
    """
    This class models rays used for raytracing that have a direction vector and magnitude.

    Attributes
    ----------
    ray_directions : torch.Tensor
        The direction of the rays, with each ray saved as a 4D vector.
    ray_magnitudes : torch.Tensor
        The magnitudes of the rays.
    """

    def __init__(self, ray_directions, ray_magnitudes) -> None:
        """
        Initialize the Rays class.

        Parameters
        ----------
        ray_directions
            The direction of the rays, with each ray saved as a 4D vector.
        ray_magnitudes : torch.Tensor
            The magnitudes of the rays.

        Raises
        ------
        AssertionError
            If the length of the ray directions does not match the length of the ray magnitudes.
        """
        assert len(ray_directions) == len(
            ray_magnitudes
        ), "Ray directions and magnitudes have differing lengths!"
        self.ray_directions = ray_directions
        self.ray_magnitudes = ray_magnitudes
