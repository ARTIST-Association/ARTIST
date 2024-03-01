"""A util script for generating scenario h5 files to be used in Artist."""

from collections.abc import MutableMapping
from typing import Dict

import h5py
from artist import ARTIST_ROOT
from artist.util.heliostat_configurations import _test_binp_heliostat

# The following configurations can be adapted to define the required scenario.

# The following parameter is the name of the scenario.
name = "test_scenario"

# The following parameters refer to the receiver.
receiver_params = {
    "center": [[0.0], [-50.0], [0.0]],
}

# The following parameters refer to the sun.
sun_params = {
    "distribution_parameters": {
        "distribution_type": str("normal"),
        "mean": 0.0,
        "covariance": 4.3681e-06,
    },
    "redraw_random_variables": False,
}

# The following parameters refer to the heliostat list.
heliostats = {
    "Single_Heliostat": {
        "id": 0,
        "aim_point": [0.0, -50.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "parameters": _test_binp_heliostat,
        "alignment_data": False,
    },
}

def flatten_dict(
    dictionary: MutableMapping, parent_key: str = "", sep: str = "/"
) -> Dict:
    """
    Flattens nested dictionaries to first level keys.

    Parameters
    ----------
    dictionary : MutableMapping
        Original nested dictionary to flatten.

    parent_key: str
        The parent key of nested dictionaries. Should be empty upon initialisation.

    sep: str
        The separator used to separate keys in nested dictionaries.

    Returns
    -------
    Dict
        A flattened version of the original dictionary
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
    Includes the parameters from the sun parameter dict.

    Parameters
    ----------
    file: h5py.File
        The h5 file to write to.

    prefix: str
        The prefix used for naming the parameters.

    parameters: dict
        The parameters to be included into the h5 file.

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
            file=f, prefix="receiver", parameters=flatten_dict(receiver_params)
        )

        # Include parameters for the sun
        include_parameters(file=f, prefix="sun", parameters=flatten_dict(sun_params))

        # Include heliostat parameters
        include_parameters(
            file=f, prefix="heliostats", parameters=flatten_dict(heliostats)
        )


if __name__ == "__main__":
    """
    The main method that generates the scenario using the parameters defined above.

    """
    generate_scenario(scenario_name=name)
