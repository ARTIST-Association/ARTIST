"""A util script for generating scenario h5 files to be used in ARTIST."""

import math
from collections.abc import MutableMapping
from typing import Dict

import h5py

from artist import ARTIST_ROOT
from artist.util import config_dictionary

# The following configurations can be adapted to define the required scenario.

# The following parameter is the name of the scenario.
name = "test_scenario"

# The following parameters refer to the receiver.
receiver_params = {
    config_dictionary.receiver_center: [0.0, -50.0, 0.0, 1.0],
}

# The following parameters refer to the sun.
sun_params = {
    config_dictionary.sun_distribution_parameters: {
        config_dictionary.sun_distribution_type: str(
            config_dictionary.sun_distribution_is_normal
        ),
        config_dictionary.sun_mean: 0.0,
        config_dictionary.sun_covariance: 4.3681e-06,
    },
    config_dictionary.sun_number_of_rays: 10,
}

# The following parameter is the name of the h5 file containing measurements that are general for a series of heliostats.
general_surface_measurements = "test_data"

# The following parameters refer to the heliostat list.
heliostats = {
    config_dictionary.general_surface_points: h5py.File(
        f"{ARTIST_ROOT}/{config_dictionary.measurement_location}/{general_surface_measurements}.h5",
        "r",
    )[config_dictionary.load_points_key][()],
    config_dictionary.general_surface_normals: h5py.File(
        f"{ARTIST_ROOT}/{config_dictionary.measurement_location}/{general_surface_measurements}.h5",
        "r",
    )[config_dictionary.load_normals_key][()],
    "Single_Heliostat": {
        config_dictionary.heliostat_id: 0,
        config_dictionary.alignment_type_key: config_dictionary.rigid_body_key,
        config_dictionary.actuator_type_key: config_dictionary.ideal_actuator_key,
        config_dictionary.heliostat_position: [0.0, 5.0, 0.0, 1.0],
        config_dictionary.heliostat_aim_point: [0.0, -50.0, 0.0, 1.0],
        config_dictionary.facets_type_key: config_dictionary.point_cloud_facet_key,
        config_dictionary.has_individual_surface_points: False,
        config_dictionary.has_individual_surface_normals: False,
        config_dictionary.heliostat_individual_surface_points: False,
        config_dictionary.heliostat_individual_surface_normals: False,
        config_dictionary.kinematic_deviation_key: {
            config_dictionary.first_joint_translation_e: 0.0,
            config_dictionary.first_joint_translation_n: 0.0,
            config_dictionary.first_joint_translation_u: 0.0,
            config_dictionary.first_joint_tilt_e: 0.0,
            config_dictionary.first_joint_tilt_n: 0.0,
            config_dictionary.first_joint_tilt_u: 0.0,
            config_dictionary.second_joint_translation_e: 0.0,
            config_dictionary.second_joint_translation_n: 0.0,
            config_dictionary.second_joint_translation_u: 0.0,
            config_dictionary.second_joint_tilt_e: 0.0,
            config_dictionary.second_joint_tilt_n: 0.0,
            config_dictionary.second_joint_tilt_u: 0.0,
            config_dictionary.concentrator_translation_e: 0.0,
            config_dictionary.concentrator_translation_n: 0.0,
            config_dictionary.concentrator_translation_u: 0.0,
            config_dictionary.concentrator_tilt_e: 0.0,
            config_dictionary.concentrator_tilt_n: 0.0,
            config_dictionary.concentrator_tilt_u: 0.0,
        },
        config_dictionary.kinematic_initial_orientation_offset_key: math.pi / 2,
    },
}


def flatten_dict(
    dictionary: MutableMapping, parent_key: str = "", sep: str = "/"
) -> Dict:
    """
    Flatten nested dictionaries to first-level keys.

    Parameters
    ----------
    dictionary : MutableMapping
        Original nested dictionary to flatten.

    parent_key : str
        The parent key of nested dictionaries. Should be empty upon initialization.

    sep : str
        The separator used to separate keys in nested dictionaries.

    Returns
    -------
    Dict
        A flattened version of the original dictionary.
    """
    return dict(_flatten_dict_gen(dictionary, parent_key, sep))


def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def include_parameters(file: h5py.File, prefix: str, parameters: dict) -> None:
    """
    Include the parameters from the sun parameter dictionary.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to write to.

    prefix : str
        The prefix used for naming the parameters.

    parameters : dict
        The parameters to be included into the hdf5 file.
    """
    for key, value in parameters.items():
        file[f"{prefix}/{key}"] = value


def generate_scenario(scenario_name: str, version: float = 0.1):
    """
    Generate the scenario according to the given parameters.

    Parameters
    ----------
    scenario_name: str
        The name of the scenario being generated.

    version : float
        The current version of the scenario generator being used. This must be updated if the names of
        the parameters are altered.

    """
    with h5py.File(f"{ARTIST_ROOT}/scenarios/{scenario_name}.h5", "w") as f:
        # Set scenario version as attribute
        f.attrs["version"] = version

        # Include parameters for the receiver
        include_parameters(
            file=f,
            prefix=config_dictionary.receiver_prefix,
            parameters=flatten_dict(receiver_params),
        )

        # Include parameters for the sun
        include_parameters(
            file=f,
            prefix=config_dictionary.sun_prefix,
            parameters=flatten_dict(sun_params),
        )

        # Include heliostat parameters
        include_parameters(
            file=f,
            prefix=config_dictionary.heliostat_prefix,
            parameters=flatten_dict(heliostats),
        )


if __name__ == "__main__":
    """
    The main method that generates the scenario using the parameters defined above.

    """
    generate_scenario(scenario_name=name)
