from typing import Any, Dict, Tuple

import pytest
import torch

from artist.scene import Sun
from artist.util import config_dictionary


def calculate_expected(
    distribution_parameters: Dict[str, Any],
    further_parameters: Dict[str, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the expected distortions given the parameters from the test fixtures.

    Parameters
    ----------
    distribution_parameters : Dict[str, Any]
        The distribution parameters for the sun.
    further_parameters : Dict[str, int]
        The further parameters for the test: number of heliostats, number of rays, number of points, and random seed.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The expected distortions in the up and east direction.
    """
    mean = torch.tensor(
        [
            distribution_parameters[config_dictionary.light_source_mean],
            distribution_parameters[config_dictionary.light_source_mean],
        ],
        dtype=torch.float,
        device=device,
    )
    covariance = torch.tensor(
        [
            [distribution_parameters[config_dictionary.light_source_covariance], 0],
            [0, distribution_parameters[config_dictionary.light_source_covariance]],
        ],
        dtype=torch.float,
        device=device,
    )
    torch.manual_seed(further_parameters["random_seed"])
    distribute = torch.distributions.MultivariateNormal(mean, covariance)
    distort_u, distort_e = distribute.sample(
        (
            int(further_parameters["num_heliostats"] * further_parameters["num_rays"]),
            further_parameters["num_facets"],
            further_parameters["num_points"],
        ),
    ).permute(3, 0, 1, 2)
    return distort_u, distort_e


@pytest.fixture
def distribution_parameters_1() -> Dict[str, Any]:
    """
    Fixture that returns distribution parameters for the sun.

    Returns
    -------
    Dict[str, Any]
        Distribution parameters for the sun.
    """
    return {
        config_dictionary.light_source_distribution_type: config_dictionary.light_source_distribution_is_normal,
        config_dictionary.light_source_mean: 0,
        config_dictionary.light_source_covariance: 1,
    }


@pytest.fixture
def distribution_parameters_2() -> Dict[str, Any]:
    """
    Fixture that returns distribution parameters for the sun.

    Returns
    -------
    Dict[str, Any]
        Distribution parameters for the sun.
    """
    return {
        config_dictionary.light_source_distribution_type: config_dictionary.light_source_distribution_is_normal,
        config_dictionary.light_source_mean: 0,
        config_dictionary.light_source_covariance: 0.004596,
    }


@pytest.fixture
def distribution_parameters_3() -> Dict[str, Any]:
    """
    Fixture that returns distribution parameters for the sun.

    Returns
    -------
    Dict[str, Any]
        Distribution parameters for the sun.
    """
    return {
        config_dictionary.light_source_distribution_type: config_dictionary.light_source_distribution_is_normal,
        config_dictionary.light_source_mean: 10,
        config_dictionary.light_source_covariance: 15,
    }


@pytest.fixture
def further_parameters_1() -> Dict[str, int]:
    """
    Fixture that returns further test parameters.

    Returns
    -------
    Dict[str, int]
        Further test parameters.
    """
    return {
        "num_rays": 100,
        "num_points": 50,
        "num_facets": 4,
        "num_heliostats": 1,
        "random_seed": 7,
    }


@pytest.fixture
def further_parameters_2() -> Dict[str, int]:
    """
    Fixture that returns further test parameters.

    Returns
    -------
    Dict[str, int]
        Further test parameters.
    """
    return {
        "num_rays": 100,
        "num_points": 50,
        "num_facets": 7,
        "num_heliostats": 5,
        "random_seed": 7,
    }


@pytest.fixture
def further_parameters_3() -> Dict[str, int]:
    """
    Fixture that returns further test parameters.

    Returns
    -------
    Dict[str, int]
        Further test parameters.
    """
    return {
        "num_rays": 20,
        "num_points": 300,
        "num_facets": 3,
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
    request: pytest.FixtureRequest,
    light_source: str,
    distribution_parameters_fixture: str,
    further_parameters_fixture: str,
    device: torch.device,
) -> None:
    """
    Test the light sources by generating distortions and ensuring these are as expected.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest request.
    light_source : str
        Indicates which light source is tested.
    distribution_parameters_fixture : str
        The pytest fixture containing the distribution parameters.
    further_parameters_fixture : str
        The pytest fixture containing the further test parameters.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
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
            device=device,
        )
        distortions_u, distortions_e = sun.get_distortions(
            number_of_points=further_params_dict["num_points"],
            number_of_heliostats=further_params_dict["num_heliostats"],
            number_of_facets=further_params_dict["num_facets"],
            random_seed=further_params_dict["random_seed"],
        )
        expected_u, expected_e = calculate_expected(
            request.getfixturevalue(distribution_parameters_fixture),
            request.getfixturevalue(further_parameters_fixture),
            device=device,
        )
        torch.testing.assert_close(distortions_u, expected_u)
        torch.testing.assert_close(distortions_e, expected_e)
