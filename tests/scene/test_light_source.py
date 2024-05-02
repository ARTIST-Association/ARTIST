from typing import Any, Dict, Tuple

import pytest
import torch

from artist.scene import Sun
from artist.util import config_dictionary


def calculate_expected(
    distribution_parameters_1: Dict[str, Any], further_parameters_1: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the expected distortions given the parameters from the test fixtures.

    Parameters
    ----------
    distribution_parameters_1 : Dict[str, Any]
        The distribution parameters for the sun.
    further_parameters_1 : Dict[str, int]
        The further parameters for the test: number of heliostats, number of rays, number of points, and random seed.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The expected distortions in the up and east direction.
    """
    mean = torch.tensor(
        [
            distribution_parameters_1[config_dictionary.sun_mean],
            distribution_parameters_1[config_dictionary.sun_mean],
        ],
        dtype=torch.float,
    )
    covariance = torch.tensor(
        [
            [distribution_parameters_1[config_dictionary.sun_covariance], 0],
            [0, distribution_parameters_1[config_dictionary.sun_covariance]],
        ],
        dtype=torch.float,
    )
    torch.manual_seed(further_parameters_1["random_seed"])
    distribute = torch.distributions.MultivariateNormal(mean, covariance)
    distort_u, distort_e = distribute.sample(
        (
            int(
                further_parameters_1["num_heliostats"]
                * further_parameters_1["num_rays"]
            ),
            further_parameters_1["num_points"],
        ),
    ).permute(2, 0, 1)
    return distort_u, distort_e


@pytest.fixture
def distribution_parameters_1() -> Dict[str, Any]:
    """Fixture that returns distribution parameters for the sun."""
    return {
        config_dictionary.sun_distribution_type: config_dictionary.sun_distribution_is_normal,
        config_dictionary.sun_mean: 0,
        config_dictionary.sun_covariance: 1,
    }


@pytest.fixture
def distribution_parameters_2() -> Dict[str, Any]:
    """Fixture that returns distribution parameters for the sun."""
    return {
        config_dictionary.sun_distribution_type: config_dictionary.sun_distribution_is_normal,
        config_dictionary.sun_mean: 0,
        config_dictionary.sun_covariance: 0.004596,
    }


@pytest.fixture
def distribution_parameters_3() -> Dict[str, Any]:
    """Fixture that returns distribution parameters for the sun."""
    return {
        config_dictionary.sun_distribution_type: config_dictionary.sun_distribution_is_normal,
        config_dictionary.sun_mean: 10,
        config_dictionary.sun_covariance: 15,
    }


@pytest.fixture
def further_parameters_1() -> Dict[str, int]:
    """Fixture that returns further test parameters."""
    return {
        "num_rays": 100,
        "num_points": 50,
        "num_heliostats": 1,
        "random_seed": 7,
    }


@pytest.fixture
def further_parameters_2() -> Dict[str, int]:
    """Fixture that returns further test parameters."""
    return {
        "num_rays": 100,
        "num_points": 50,
        "num_heliostats": 5,
        "random_seed": 7,
    }


@pytest.fixture
def further_parameters_3() -> Dict[str, int]:
    """Fixture that returns further test parameters."""
    return {
        "num_rays": 20,
        "num_points": 300,
        "num_heliostats": 8,
        "random_seed": 77,
    }


@pytest.mark.parametrize(
    "light_source, distribution_parameters_fixture, further_parameters_fixture",
    [
        ("sun", "distribution_parameters_1", "further_parameters_1"),
        ("sun", "distribution_parameters_2", "further_parameters_2"),
        ("sun", "distribution_parameters_3", "further_parameters_3"),
        ("sun", "distribution_parameters_1", "further_parameters_2"),
        ("sun", "distribution_parameters_1", "further_parameters_3"),
        ("sun", "distribution_parameters_2", "further_parameters_1"),
        ("sun", "distribution_parameters_2", "further_parameters_3"),
        ("sun", "distribution_parameters_3", "further_parameters_1"),
        ("sun", "distribution_parameters_3", "further_parameters_2"),
    ],
)
def test_light_sources(
    request: Any,
    light_source: str,
    distribution_parameters_fixture: Dict[str, Any],
    further_parameters_fixture: Dict[str, int],
) -> None:
    """
    Test the light sources by generating distortions and ensuring these are as expected.

    Parameters
    ----------
    request : Any
        The pytest request.
    light_source : str
        Indicates which light source is tested.
    distribution_parameters_fixture : Dict[str, Any]
        The pytest fixture containing the distribution parameters.
    further_parameters_fixture : Dict[str, int]
        The pytest fixture containing the further test parameters.
    """
    # Load further params dict.
    further_params_dict = request.getfixturevalue(further_parameters_fixture)

    # Run test if light source is a sun.
    if light_source == "sun":
        sun = Sun(
            distribution_parameters=request.getfixturevalue(
                distribution_parameters_fixture
            ),
            number_of_rays=further_params_dict["num_rays"],
        )
        distortions_u, distortions_e = sun.get_distortions(
            number_of_points=further_params_dict["num_points"],
            number_of_heliostats=further_params_dict["num_heliostats"],
            random_seed=further_params_dict["random_seed"],
        )
        expected_u, expected_e = calculate_expected(
            request.getfixturevalue(distribution_parameters_fixture),
            request.getfixturevalue(further_parameters_fixture),
        )
        torch.testing.assert_close(distortions_u, expected_u)
        torch.testing.assert_close(distortions_e, expected_e)
