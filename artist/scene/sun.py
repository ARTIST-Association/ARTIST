import logging
from typing import Any, Optional, Union

import h5py
import torch
from typing_extensions import Self

from artist.scene.light_source import LightSource
from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the sun."""


class Sun(LightSource):
    """
    Implement the sun as a light source.

    Attributes
    ----------
    number_of_rays : int
        The number of sent-out rays sampled from the sun distribution.
    distribution_parameters : dict[str, Any]
        Parameters of the distribution used to model the sun.

    Methods
    -------
    from_hdf5()
        Class method to initialize a sun from an HDF5 file.
    get_distortions()
        Returns distortions used to rotate rays.
    forward()
        Specify the forward pass.

    See Also
    --------
    :class:`LightSource`: Reference to the parent class.
    """

    def __init__(
        self,
        number_of_rays: int,
        distribution_parameters: dict[str, Any] = dict(
            distribution_type="normal", mean=0.0, covariance=4.3681e-06
        ),
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the sun as a light source.

        The sun is one type of light source that can be implemented in ``ARTIST``. The number of rays sent out by the
        light source per heliostat surface point must be specified. If more rays are sent out, the resulting flux
        density distribution on the receiver is higher. Furthermore, each light source also implements the
        ``get_distortions`` function required to scatter the light.

        Parameters
        ----------
        number_of_rays : int
            The number of sent-out rays sampled from the sun distribution.
        distribution_parameters
            Parameters of the distribution used to model the sun.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        ValueError
            If the specified distribution type is unknown.
        """
        super().__init__(number_of_rays=number_of_rays)
        device = torch.device(device)

        self.distribution_parameters = distribution_parameters
        self.number_of_rays = number_of_rays
        if (
            self.distribution_parameters[
                config_dictionary.light_source_distribution_type
            ]
            != config_dictionary.light_source_distribution_is_normal
        ):
            raise ValueError("Unknown sunlight distribution type.")

        if (
            self.distribution_parameters[
                config_dictionary.light_source_distribution_type
            ]
            == config_dictionary.light_source_distribution_is_normal
        ):
            log.info(
                "Initializing a sun modeled with a multivariate normal distribution."
            )
            mean = torch.tensor(
                [
                    self.distribution_parameters[config_dictionary.light_source_mean],
                    self.distribution_parameters[config_dictionary.light_source_mean],
                ],
                dtype=torch.float,
                device=device,
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
                device=device,
            )

            self.distribution = torch.distributions.MultivariateNormal(mean, covariance)

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        light_source_name: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Class method that initializes a sun from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the sun.
        light_source_name : str, optional
            The name of the light source - used for logging.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Sun
            A sun initialized from an HDF5 file.
        """
        if light_source_name:
            log.info(f"Loading {light_source_name} from an HDF5 file.")
        device = torch.device(device)

        number_of_rays = int(
            config_file[config_dictionary.light_source_number_of_rays][()]
        )

        distribution_parameters = {
            config_dictionary.light_source_distribution_type: config_file[
                config_dictionary.light_source_distribution_parameters
            ][config_dictionary.light_source_distribution_type][()].decode("utf-8")
        }

        if (
            config_dictionary.light_source_mean
            in config_file[
                config_dictionary.light_source_distribution_parameters
            ].keys()
        ):
            distribution_parameters.update(
                {
                    config_dictionary.light_source_mean: float(
                        config_file[
                            config_dictionary.light_source_distribution_parameters
                        ][config_dictionary.light_source_mean][()]
                    )
                }
            )

        if (
            config_dictionary.light_source_covariance
            in config_file[
                config_dictionary.light_source_distribution_parameters
            ].keys()
        ):
            distribution_parameters.update(
                {
                    config_dictionary.light_source_covariance: float(
                        config_file[
                            config_dictionary.light_source_distribution_parameters
                        ][config_dictionary.light_source_covariance][()]
                    )
                }
            )

        return cls(
            number_of_rays=number_of_rays,
            distribution_parameters=distribution_parameters,
            device=device,
        )

    def get_distortions(
        self,
        number_of_points: int,
        number_of_facets: int = 4,
        number_of_heliostats: int = 1,
        random_seed: int = 7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the selected model of the sun.

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

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The distortion in north and up direction.

        Raises
        ------
        ValueError
            If the distribution type is not valid, currently only the normal distribution is implemented.
        """
        torch.manual_seed(random_seed)
        if (
            self.distribution_parameters[
                config_dictionary.light_source_distribution_type
            ]
            == config_dictionary.light_source_distribution_is_normal
        ):
            distortions_u, distortions_e = self.distribution.sample(
                (
                    int(number_of_heliostats * self.number_of_rays),
                    number_of_facets,
                    number_of_points,
                ),
            ).permute(3, 0, 1, 2)
            return distortions_u, distortions_e
        else:
            raise ValueError("Unknown light distribution type.")

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")