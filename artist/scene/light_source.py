import torch
from typing_extensions import Self


class LightSource(torch.nn.Module):
    """
    Abstract base class for all light sources.

    Attributes
    ----------
    number_of_rays : int
        The number of sent-out rays sampled from the light source distribution.

    Methods
    -------
    get_distortions()
        Get distortions given the light source model.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        number_of_rays: int,
    ) -> None:
        """
        Initialize the light source.

        The abstract light source implements a template for the construction of light sources
        which can be of various types.
        The most noticeable light source is the sun, however, drones could carry artificial light sources and
        distribute this light on specific heliostats for calibration purposes.
        Currently, only the sun is implemented.

        Parameters
        ----------
        number_of_rays : int
            The number of sent-out rays sampled from the sun distribution.
        """
        super().__init__()
        self.number_of_rays = number_of_rays

    @classmethod
    def from_hdf5(
        cls,
    ) -> Self:
        """
        Load the light source from an HDF5 file.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def get_distortions(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the light source model.

        This function gets the distortions that are later used to model possible ray directions that are being generated
        from the light source. Depending on the model of the light source, the distortions are generated differently.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Must be overridden!")
