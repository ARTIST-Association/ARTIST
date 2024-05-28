from typing import Optional, Tuple

import h5py
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
    """

    def __init__(
        self,
        number_of_rays: int,
    ) -> None:
        """
        Initialize the light source.

        The abstract light source implements a template for the construction of inheriting light sources
        which can be of various types. The most apparent light source is the sun, however also drones could
        carry artificial light sources and shine light on specific heliostats for calibration purposes.
        Currently only the sun is implemented.

        Parameters
        ----------
        number_of_rays : int
            The number of sent-out rays sampled from the sun distribution.
        """
        super().__init__()
        self.number_of_rays = number_of_rays

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, light_source_name: Optional[str] = None
    ) -> Self:
        """
        Load the light source from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the light source.
        light_source_name : str, optional
            The name of the light source - used for logging.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def get_distortions(
        self,
        number_of_points: int,
        number_of_facets: int = 4,
        number_of_heliostats: int = 1,
        random_seed: int = 7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the light source model.

        This function gets the distortions that are later used to model possible rays that are being generated
        from the light source. Depending on the model of the sun, the distortions are generated differently.

        Parameters
        ----------
        number_of_points : int
            The number of points on the heliostat.
        number_of_facets : int, optional
            The number of facets per heliostat (default: 4).
        number_of_heliostats : int, optional
            The number of heliostats in the scenario (default: 1).
        random_seed : int
            The random seed to enable result replication (default: 7).

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
