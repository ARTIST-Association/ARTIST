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
    number_of_rays : int
        The number of sent-out rays sampled from the sun distribution.
    distribution_parameters : Dict[str, Any]
        Parameters of the distribution used to model the sun.

    Methods
    -------
    from_hdf5()
        Class method to initialize a sun from an HDF5 file.
    get_distortions()
        Returns distortions used to rotate rays.

    See Also
    --------
    :class:`LightSource` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_rays: int,
        distribution_parameters: Dict[str, Any] = dict(
            distribution_type="normal", mean=0.0, covariance=4.3681e-06
        ),
    ) -> None:
        """
        Initialize the sun as a light source.

        Parameters
        ----------
        number_of_rays : int
            The number of sent-out rays sampled from the sun distribution.
        distribution_parameters
            Parameters of the distribution used to model the sun.

        Raises
        ------
        Union[ValueError, NotImplementedError]
            If the specified distribution type is unknown.
        """
        super().__init__(number_of_rays=number_of_rays)

        self.distribution_parameters = distribution_parameters
        self.number_of_rays = number_of_rays

        assert (
            self.distribution_parameters[
                config_dictionary.light_source_distribution_type
            ]
            == config_dictionary.light_source_distribution_is_normal
        ), "Unknown sunlight distribution type."

        if (
            self.distribution_parameters[
                config_dictionary.light_source_distribution_type
            ]
            == config_dictionary.light_source_distribution_is_normal
        ):
            mean = torch.tensor(
                [
                    self.distribution_parameters[config_dictionary.light_source_mean],
                    self.distribution_parameters[config_dictionary.light_source_mean],
                ],
                dtype=torch.float,
            )
            covariance = torch.tensor(
                [
                    [
                        self.distribution_parameters[
                            config_dictionary.light_source_covariance
                        ],
                        0,
                    ],
                    [
                        0,
                        self.distribution_parameters[
                            config_dictionary.light_source_covariance
                        ],
                    ],
                ],
                dtype=torch.float,
            )

            self.distribution = torch.distributions.MultivariateNormal(mean, covariance)

    @classmethod
    def from_hdf5(cls, config_file: h5py.File, light_source_key: str) -> Self:
        """
        Class method that initializes a sun from an hdf5 file.

        Parameters
        ----------
        config_file : h5py.File
            The hdf5 file containing the information about the sun.
        light_source_key : str
            The key identifying the light source to be loaded.

        Returns
        -------
        Sun
            A sun initialized from an hdf5 file.
        """
        number_of_rays = int(
            config_file[config_dictionary.light_source_key][light_source_key][
                config_dictionary.light_source_number_of_rays
            ][()]
        )

        distribution_parameters = {
            config_dictionary.light_source_distribution_type: config_file[
                config_dictionary.light_source_key
            ][light_source_key][config_dictionary.light_source_distribution_parameters][
                config_dictionary.light_source_distribution_type
            ][()].decode("utf-8")
        }

        if (
            config_dictionary.light_source_mean
            in config_file[config_dictionary.light_source_key][light_source_key][
                config_dictionary.light_source_distribution_parameters
            ].keys()
        ):
            distribution_parameters.update(
                {
                    config_dictionary.light_source_mean: float(
                        config_file[config_dictionary.light_source_key][
                            light_source_key
                        ][config_dictionary.light_source_distribution_parameters][
                            config_dictionary.light_source_mean
                        ][()]
                    )
                }
            )

        if (
            config_dictionary.light_source_covariance
            in config_file[config_dictionary.light_source_key][light_source_key][
                config_dictionary.light_source_distribution_parameters
            ].keys()
        ):
            distribution_parameters.update(
                {
                    config_dictionary.light_source_covariance: float(
                        config_file[config_dictionary.light_source_key][
                            light_source_key
                        ][config_dictionary.light_source_distribution_parameters][
                            config_dictionary.light_source_covariance
                        ][()]
                    )
                }
            )

        return cls(
            number_of_rays=number_of_rays,
            distribution_parameters=distribution_parameters,
        )

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
                (int(number_of_heliostats * self.number_of_rays), number_of_points),
            ).permute(2, 0, 1)
            return distortions_u, distortions_e
        else:
            raise ValueError("Unknown light distribution type.")
