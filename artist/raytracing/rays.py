import torch


class Rays:
    """
    Model rays used for raytracing that have a direction vector and magnitude.

    Attributes
    ----------
    ray_directions : torch.Tensor
        The direction of the rays, with each ray saved as a 4D vector.
    ray_magnitudes : torch.Tensor
        The magnitudes of the rays.
    """

    def __init__(
        self, ray_directions: torch.Tensor, ray_magnitudes: torch.Tensor
    ) -> None:
        """
        Initialize the ``Rays`` class.

        The rays in ``ARTIST`` have a direction vector and a magnitude. They are used for raytracing.
        The direction vector determines the direction of the rays, i.e., the path they are taking through space.
        The magnitude is important for considering atmospheric losses and cloud coverage. If a ray
        travels through a cloud, the magnitude changes.

        Parameters
        ----------
        ray_directions : torch.Tensor
            The direction of the rays, with each ray saved as a 4D vector.
        ray_magnitudes : torch.Tensor
            The magnitudes of the rays.

        Raises
        ------
        AssertionError
            If the length of the ray directions does not match the length of the ray magnitudes.
        """
        assert (
            ray_directions.size(dim=0) == ray_magnitudes.size(dim=0)
            and ray_directions.size(dim=1) == ray_magnitudes.size(dim=1)
            and ray_directions.size(dim=2) == ray_magnitudes.size(dim=2)
        ), "Ray directions and magnitudes have differing sizes!"
        self.ray_directions = ray_directions
        self.ray_magnitudes = ray_magnitudes
