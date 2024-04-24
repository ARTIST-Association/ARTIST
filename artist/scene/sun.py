from typing import Any, Dict, Optional, Tuple

import h5py
import torch
from typing_extensions import Self

from artist.scene.light_source import LightSource
from artist.util import config_dictionary


class Sun(LightSource):
    """
    This class implements the sun as a light source.

    Attributes
    ----------
    distribution_parameters : Dict[str, Any]
        Parameters of the distribution used to model the sun.
    ray_count : int
        The number of sent-out rays sampled from the sun distribution.

    Methods
    -------
    from_hdf5()
        Class method to initialize heliostat from a h5 file.
    get_distortions()
        Returns distortions used to rotate rays.

    See Also
    --------
    :class:`LightSource` : Reference to the parent class.
    """

    def __init__(
        self,
        distribution_parameters: Dict[str, Any] = dict(
            distribution_type="normal", mean=0.0, covariance=4.3681e-06
        ),
        ray_count: int = 200,
    ) -> None:
        """
        Initialize the sun as a light source.

        Parameters
        ----------
        distribution_parameters
            Parameters of the distribution used to model the sun.
        ray_count : int
            The number of sent-out rays sampled from the sun distribution.

        Raises
        ------
        Union[ValueError, NotImplementedError]
            If the specified distribution type is unknown.
        """
        super().__init__()

        self.distribution_parameters = distribution_parameters
        self.ray_count = ray_count

        assert (
            self.distribution_parameters[config_dictionary.sun_distribution_type]
            == config_dictionary.sun_distribution_is_normal
        ), "Unknown sunlight distribution type."

        if (
            self.distribution_parameters[config_dictionary.sun_distribution_type]
            == config_dictionary.sun_distribution_is_normal
        ):
            mean = torch.tensor(
                [
                    self.distribution_parameters[config_dictionary.sun_mean],
                    self.distribution_parameters[config_dictionary.sun_mean],
                ],
                dtype=torch.float,
            )
            covariance = torch.tensor(
                [
                    [self.distribution_parameters[config_dictionary.sun_covariance], 0],
                    [0, self.distribution_parameters[config_dictionary.sun_covariance]],
                ],
                dtype=torch.float,
            )

            self.distribution = torch.distributions.MultivariateNormal(mean, covariance)

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Class method that initializes a sun from an hdf5 file.

        Parameters
        ----------
        config_file : h5py.File
            The hdf5 file containing the information about the sun.

        Returns
        -------
        Sun
            A sun initialized from an hdf5 file.
        """
        distribution_parameters = {
            config_dictionary.sun_distribution_type: config_file[
                config_dictionary.sun_prefix
            ][config_dictionary.sun_distribution_parameters][
                config_dictionary.sun_distribution_type
            ][()].decode("utf-8"),
            config_dictionary.sun_mean: config_file[config_dictionary.sun_prefix][
                config_dictionary.sun_distribution_parameters
            ][config_dictionary.sun_mean][()],
            config_dictionary.sun_covariance: config_file[config_dictionary.sun_prefix][
                config_dictionary.sun_distribution_parameters
            ][config_dictionary.sun_covariance][()],
        }
        num_rays = config_file[config_dictionary.sun_prefix][
            config_dictionary.sun_number_of_rays
        ][()]

        return cls(distribution_parameters=distribution_parameters, ray_count=num_rays)

    def get_distortions(
        self,
        number_of_points: int,
        number_of_heliostats: Optional[int] = 1,
        random_seed: Optional[int] = 7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the selected model of the sun.

        Parameters
        ----------
        number_of_points : int
            The number of points on the heliostat from which rays are reflected.
        number_of_heliostats : Optional[int]
            The number of heliostats in the scenario.
        random_seed : Optional[int]
            The random seed to enable result replication.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The distortion in north and up direction.

        Raises
        ------
        ValueError
            If the distribution type is not valid, currently only the normal distribution is implemented.
        """
        torch.manual_seed(random_seed)
        if (
            self.distribution_parameters[config_dictionary.sun_distribution_type]
            == config_dictionary.sun_distribution_is_normal
        ):
            distortions_u, distortions_e = self.distribution.sample(
                (int(number_of_heliostats * self.ray_count), number_of_points),
            ).permute(2, 0, 1)
            return distortions_u, distortions_e
        else:
            raise ValueError("Unknown light distribution type.")
