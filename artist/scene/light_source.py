import h5py
import torch
from typing_extensions import Self


class LightSource:
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
        self.number_of_rays = number_of_rays

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        light_source_name: str | None = None,
        device: torch.device | None = None,
    ) -> Self:
        """
        Load the light source from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the light sources.
        light_source_name : str | None
            The name of the light source - used for logging.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def get_distortions(
        self,
        number_of_points: int,
        number_of_heliostats: int,
        random_seed: int = 7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the light source model.

        This function gets the distortions that are later used to model possible ray directions that are being generated
        from the light source. Depending on the model of the light source, the distortions are generated differently.

        Parameters
        ----------
        number_of_points : int
            The number of points on the heliostat from which rays are reflected.
        number_of_facets : int
            The number of facets for each heliostat (default: 4).
        number_of_heliostats : int
            The number of heliostats in the scenario (default: 1).
        random_seed : int
            The random seed to enable result replication (default: 7).

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
