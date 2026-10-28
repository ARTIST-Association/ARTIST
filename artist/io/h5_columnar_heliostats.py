import pathlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import torch

from artist.geometry import rotations
from artist.util import constants, indices

layout_attribute = "layout"
"""Attribute of the heliostat group in a scenario file that names the layout of the heliostats."""

columnar_layout = "columnar"
"""Layout storing each heliostat parameter of all heliostats in one dataset, see ``heliostat_columns()``."""

heliostat_indices_key = "heliostat_indices"
"""Key for the indices of the heliostats that have individual parameters for a component."""

actuator_names_key = "names"
"""Key for the actuator names of all heliostats with individual actuators."""

required_fields = {
    "": (constants.heliostat_id, constants.heliostat_position),
    constants.heliostat_surface_key: (
        constants.facet_control_points,
        constants.facet_degrees,
        constants.facets_translation_vector,
        constants.facets_canting,
    ),
    constants.heliostat_kinematics_key: (
        constants.kinematics_type,
        constants.kinematics_initial_orientation,
    ),
    constants.heliostat_actuator_key: (
        constants.actuator_type_key,
        constants.actuator_clockwise_axis_movement,
        constants.actuator_min_max_motor_positions,
    ),
}
"""The parameters every heliostat, facet, kinematics, and actuator must have, per component."""

optional_fields = {
    constants.heliostat_kinematics_key: constants.kinematics_deviations,
    constants.heliostat_actuator_key: constants.actuator_parameters_key,
}
"""The group of scalar parameters per component that may be missing, which are stored as NaN."""

string_fields = (constants.kinematics_type, constants.actuator_type_key)
"""The parameters that are stored as strings."""

rigid_body_deviations = (
    (
        (indices.first_joint_translation_e, constants.first_joint_translation_e),
        (indices.first_joint_translation_n, constants.first_joint_translation_n),
        (indices.first_joint_translation_u, constants.first_joint_translation_u),
        (indices.second_joint_translation_e, constants.second_joint_translation_e),
        (indices.second_joint_translation_n, constants.second_joint_translation_n),
        (indices.second_joint_translation_u, constants.second_joint_translation_u),
        (indices.concentrator_translation_e, constants.concentrator_translation_e),
        (indices.concentrator_translation_n, constants.concentrator_translation_n),
        (indices.concentrator_translation_u, constants.concentrator_translation_u),
    ),
    (
        (indices.first_joint_tilt_n, constants.first_joint_tilt_n),
        (indices.first_joint_tilt_u, constants.first_joint_tilt_u),
        (indices.second_joint_tilt_e, constants.second_joint_tilt_e),
        (indices.second_joint_tilt_n, constants.second_joint_tilt_n),
    ),
)
"""Index and name of each translation and each rotation deviation parameter of the rigid body kinematics."""

linear_actuator_parameters = (
    (
        (indices.actuator_increment, constants.actuator_increment),
        (indices.actuator_offset, constants.actuator_offset),
        (indices.actuator_pivot_radius, constants.actuator_pivot_radius),
    ),
    (
        (indices.actuator_initial_angle, constants.actuator_initial_angle),
        (
            indices.actuator_initial_stroke_length,
            constants.actuator_initial_stroke_length,
        ),
    ),
)
"""Index and name of each non-optimizable and each optimizable linear actuator parameter."""


class ColumnarLayoutError(ValueError):
    """Raised if heliostats cannot be stored in the columnar layout, e.g., because their parameters differ in shape."""


def is_columnar(heliostats_config: h5py.Group) -> bool:
    """
    Check whether the heliostats of a scenario file are stored in the columnar layout.

    Parameters
    ----------
    heliostats_config : h5py.Group
        The heliostat group of the scenario file.

    Returns
    -------
    bool
        Whether the heliostats are stored in the columnar layout.
    """
    return heliostats_config.attrs.get(layout_attribute) == columnar_layout


def heliostat_columns(
    heliostats: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, np.ndarray]:
    """
    Arrange the parameters of all heliostats in the columnar layout.

    Every heliostat parameter is stored in one dataset for all heliostats, with the heliostats as the first
    dimension, sorted by name. Facets and actuators are sorted by name as well, as HDF5 sorts the members of a
    group. Individual surfaces, kinematics, and actuators are only stored for the heliostats that have them,
    together with the indices of these heliostats. Missing kinematics deviations and actuator parameters are
    stored as NaN.

    Parameters
    ----------
    heliostats : Sequence[tuple[str, dict[str, Any]]]
        The name and the flattened parameters of each heliostat, as stored in the legacy layout, e.g.,
        ``{"position": ..., "surface/facets/facet_1/control_points": ...}``.

    Raises
    ------
    ColumnarLayoutError
        If the heliostats cannot be stored in the columnar layout. Nothing is silently left out: parameters
        that differ in shape between heliostats, or that the layout does not know, raise this error.

    Returns
    -------
    dict[str, np.ndarray]
        The path of each dataset relative to the heliostat group and its data.
    """
    heliostats = sorted(heliostats, key=lambda heliostat: heliostat[0])
    names = [name for name, _ in heliostats]
    if not names:
        raise ColumnarLayoutError("The scenario contains no heliostats.")
    if len(set(names)) != len(names):
        raise ColumnarLayoutError("Heliostat names must be unique.")

    # Per component, the index, name, and the parameters of each item (facet or actuator) of every heliostat
    # that has the component. Heliostat-level parameters and the kinematics are a single unnamed item.
    rows: dict[str, list[tuple[int, str, dict[str, dict[str, Any]]]]] = {
        component: [] for component in required_fields
    }
    for heliostat_index, (name, parameters) in enumerate(heliostats):
        items_per_component: dict[str, defaultdict[str, dict[str, Any]]] = {
            component: defaultdict(dict) for component in required_fields
        }
        for key, value in parameters.items():
            component, _, rest = key.partition("/") if "/" in key else ("", "", key)
            if component == constants.heliostat_surface_key:
                facets, _, rest = rest.partition("/")
                item, _, field = (
                    rest.partition("/")
                    if facets == constants.facets_key
                    else ("", "", "")
                )
            elif component == constants.heliostat_actuator_key:
                item, _, field = rest.partition("/")
            else:
                item, field = "", rest
            if component not in required_fields or not field:
                raise ColumnarLayoutError(
                    f"Heliostat {name} contains the parameter {key}, which the columnar layout cannot store."
                )
            items_per_component[component][item][field] = value
        for component, items in items_per_component.items():
            if items or not component:
                rows[component].append(
                    (heliostat_index, name, dict(sorted(items.items())) or {"": {}})
                )

    columns = {constants.names: np.array(names, dtype=h5py.string_dtype())}
    for component, component_rows in rows.items():
        if component_rows:
            columns.update(_component_columns(component, component_rows))
    return columns


def _component_columns(
    component: str, rows: list[tuple[int, str, dict[str, dict[str, Any]]]]
) -> dict[str, np.ndarray]:
    """
    Arrange the parameters of one component of all heliostats that have it in the columnar layout.

    Parameters
    ----------
    component : str
        The component, i.e., ``""`` for heliostat-level parameters, surface, kinematics, or actuators.
    rows : list[tuple[int, str, dict[str, dict[str, Any]]]]
        The index, name, and the parameters of each item (facet or actuator) of every heliostat with the
        component.

    Raises
    ------
    ColumnarLayoutError
        If a parameter is missing, unknown, not scalar where it must be, or differs in shape between heliostats.

    Returns
    -------
    dict[str, np.ndarray]
        The path of each dataset relative to the heliostat group and its data.
    """
    prefix = f"{component}/" if component else ""
    has_items = component in (
        constants.heliostat_surface_key,
        constants.heliostat_actuator_key,
    )
    columns = {}
    if component:
        columns[f"{prefix}{heliostat_indices_key}"] = np.array(
            [heliostat_index for heliostat_index, _, _ in rows], dtype=np.int64
        )
    if component == constants.heliostat_actuator_key:
        if len({len(items) for _, _, items in rows}) > 1:
            raise ColumnarLayoutError(
                "Heliostats have different numbers of actuators, so the columnar layout cannot store them."
            )
        columns[f"{prefix}{actuator_names_key}"] = np.array(
            [list(items) for _, _, items in rows], dtype=h5py.string_dtype()
        )

    for field in required_fields[component]:
        values = []
        for _, name, items in rows:
            if any(field not in parameters for parameters in items.values()):
                raise ColumnarLayoutError(
                    f"Heliostat {name} has no {prefix}{field}, which the columnar layout requires."
                )
            item_values = [
                _to_str(parameters[field])
                if field in string_fields
                else _to_numpy(parameters[field])
                for parameters in items.values()
            ]
            values.append(item_values if has_items else item_values[0])
        try:
            columns[f"{prefix}{field}"] = (
                np.array(values, dtype=h5py.string_dtype())
                if field in string_fields
                else np.stack(
                    [np.stack(value) if has_items else value for value in values]
                )
            )
        except ValueError as error:
            raise ColumnarLayoutError(
                f"The {prefix}{field} differs in shape between heliostats or their facets or actuators, "
                f"so the columnar layout cannot store it."
            ) from error

    optional = f"{optional_fields.get(component)}/"
    optional_keys = set()
    for _, name, items in rows:
        for parameters in items.values():
            for field in parameters:
                if field.startswith(optional):
                    optional_keys.add(field)
                elif field not in required_fields[component]:
                    raise ColumnarLayoutError(
                        f"Heliostat {name} contains the parameter {prefix}{field}, which the columnar layout "
                        f"cannot store."
                    )
    for key in sorted(optional_keys):
        scalars = []
        for _, name, items in rows:
            arrays = [
                _to_numpy(parameters.get(key, np.nan)) for parameters in items.values()
            ]
            if any(array.size != 1 for array in arrays):
                raise ColumnarLayoutError(
                    f"Heliostat {name} has a non-scalar {prefix}{key}, which the columnar layout cannot store."
                )
            scalars.append([float(array.reshape(())) for array in arrays])
        column = np.array(scalars, dtype=np.float64)
        columns[f"{prefix}{key}"] = column if has_items else column[:, 0]
    return columns


def _to_numpy(value: Any) -> np.ndarray:
    """
    Convert a parameter value, i.e., a tensor, NumPy array, or Python scalar, to a NumPy array.

    Parameters
    ----------
    value : Any
        The value.

    Returns
    -------
    np.ndarray
        The value as a NumPy array.
    """
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _to_str(value: Any) -> str:
    """
    Convert a string parameter value, which may be stored as bytes or a NumPy scalar, to a string.

    Parameters
    ----------
    value : Any
        The value.

    Returns
    -------
    str
        The value as a string.
    """
    value = value.item() if isinstance(value, np.ndarray | np.generic) else value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_columnar_heliostats(
    heliostats_config: h5py.Group, columns: dict[str, np.ndarray]
) -> None:
    """
    Write heliostats arranged by ``heliostat_columns()`` into an empty HDF5 group and mark its layout.

    Parameters
    ----------
    heliostats_config : h5py.Group
        The empty heliostat group of the scenario file.
    columns : dict[str, np.ndarray]
        The path of each dataset relative to the heliostat group and its data.
    """
    for path, data in columns.items():
        heliostats_config.create_dataset(path, data=data)
    heliostats_config.attrs[layout_attribute] = columnar_layout


def read_legacy_heliostats(
    heliostats_config: h5py.Group,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Read all heliostats stored in the legacy layout, with one HDF5 group per heliostat, as flattened parameters.

    Parameters
    ----------
    heliostats_config : h5py.Group
        The heliostat group of the scenario file.

    Returns
    -------
    list[tuple[str, dict[str, Any]]]
        The name and the flattened parameters of each heliostat.
    """
    heliostats = []
    for name, heliostat_config in heliostats_config.items():
        parameters: dict[str, Any] = {}
        heliostat_config.visititems(
            lambda key, item, parameters=parameters: (
                parameters.__setitem__(key, item[()])
                if isinstance(item, h5py.Dataset)
                else None
            )
        )
        heliostats.append((name, parameters))
    return heliostats


def read_heliostat_columns(
    heliostats_config: h5py.Group,
) -> list[dict[str, np.ndarray]]:
    """
    Read the heliostats of a scenario file in the columnar layout, whichever layout the file uses.

    Heliostats in the legacy layout are arranged in columns while reading. If their parameters differ in shape,
    e.g., because their surfaces have different numbers of control points, each heliostat is arranged on its own.

    Parameters
    ----------
    heliostats_config : h5py.Group
        The heliostat group of the scenario file.

    Raises
    ------
    ColumnarLayoutError
        If a heliostat in the legacy layout contains parameters that are missing or unknown.

    Returns
    -------
    list[dict[str, np.ndarray]]
        One or more sets of columns, each mapping the path of a dataset relative to the heliostat group to its
        data. Together they contain all heliostats, sorted by name.
    """
    if is_columnar(heliostats_config):
        columns: dict[str, np.ndarray] = {}
        heliostats_config.visititems(
            lambda path, item: (
                columns.__setitem__(
                    path,
                    item.asstr()[()]
                    if h5py.check_string_dtype(item.dtype)
                    else item[()],
                )
                if isinstance(item, h5py.Dataset)
                else None
            )
        )
        return [columns]

    heliostats = read_legacy_heliostats(heliostats_config)
    try:
        return [heliostat_columns(heliostats)]
    except ColumnarLayoutError:
        return [
            heliostat_columns([heliostat])
            for heliostat in sorted(heliostats, key=lambda heliostat: heliostat[0])
        ]


def read_kinematics_deviations(
    columns: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Read the rigid body deviations of all heliostats with individual kinematics onto the CPU.

    Missing deviations default to zero.

    Parameters
    ----------
    columns : dict[str, np.ndarray]
        The heliostat columns.

    Returns
    -------
    torch.Tensor
        The translation deviations.
        Shape is ``[number_of_heliostats_with_kinematics, 9]``.
    torch.Tensor
        The rotation deviations.
        Shape is ``[number_of_heliostats_with_kinematics, 4]``.
    dict[str, torch.Tensor]
        For each deviation, which heliostats with individual kinematics miss it.
        Shape is ``[number_of_heliostats_with_kinematics]``.
    """
    prefix = f"{constants.heliostat_kinematics_key}/{constants.kinematics_deviations}/"
    number_of_rows = len(
        columns[f"{constants.heliostat_kinematics_key}/{heliostat_indices_key}"]
    )
    deviations, missing = [], {}
    for parameters in rigid_body_deviations:
        values = torch.zeros((number_of_rows, len(parameters)))
        for index, key in parameters:
            column = torch.as_tensor(
                columns.get(f"{prefix}{key}", np.full(number_of_rows, np.nan))
            )
            missing[key] = column.isnan()
            values[:, index] = column.nan_to_num(0.0).float()
        deviations.append(values)
    return deviations[0], deviations[1], missing


def read_actuator_parameters(
    columns: dict[str, np.ndarray], actuator_type: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Read the parameters of all heliostats with individual actuators onto the CPU, as actuators of one type.

    Missing parameters default to zero. For linear actuators, the initial angle of the first actuator is
    corrected for the orientation of the surface, as in ``h5_scenario_parser.linear_actuators()``.

    Parameters
    ----------
    columns : dict[str, np.ndarray]
        The heliostat columns.
    actuator_type : str
        The actuator type.

    Raises
    ------
    ValueError
        If the actuator type is unknown.

    Returns
    -------
    torch.Tensor
        The non-optimizable parameters.
        Shape is ``[number_of_heliostats_with_actuators, number_of_parameters, number_of_actuators]``.
    torch.Tensor
        The optimizable parameters.
        Shape is ``[number_of_heliostats_with_actuators, 2, number_of_actuators]`` for linear actuators.
    dict[str, torch.Tensor]
        For each linear actuator parameter, which actuators of the heliostats miss it.
        Shape is ``[number_of_heliostats_with_actuators, number_of_actuators]``.
    """
    prefix = f"{constants.heliostat_actuator_key}/"
    clockwise_axis_movement = torch.as_tensor(
        columns[f"{prefix}{constants.actuator_clockwise_axis_movement}"]
    )
    min_max_motor_positions = torch.as_tensor(
        columns[f"{prefix}{constants.actuator_min_max_motor_positions}"]
    ).double()
    number_of_rows, number_of_actuators = clockwise_axis_movement.shape

    if actuator_type == constants.ideal_actuator_key:
        actuator_type_int, number_of_non_optimizable_parameters = (
            constants.ideal_actuator_int,
            4,
        )
    elif actuator_type == constants.linear_actuator_key:
        actuator_type_int, number_of_non_optimizable_parameters = (
            constants.linear_actuator_int,
            7,
        )
    else:
        raise ValueError(f"The actuator type: {actuator_type} is not yet implemented!")

    non_optimizable = torch.zeros(
        (number_of_rows, number_of_non_optimizable_parameters, number_of_actuators)
    )
    non_optimizable[:, indices.actuator_type] = actuator_type_int
    non_optimizable[:, indices.actuator_clockwise_movement] = (
        clockwise_axis_movement.float()
    )
    non_optimizable[:, indices.actuator_min_motor_position] = min_max_motor_positions[
        ..., indices.data_actuator_min_motor_position
    ].float()
    non_optimizable[:, indices.actuator_max_motor_position] = min_max_motor_positions[
        ..., indices.data_actuator_max_motor_position
    ].float()
    if actuator_type == constants.ideal_actuator_key:
        return non_optimizable, torch.empty((number_of_rows, 0)), {}

    optimizable = torch.zeros((number_of_rows, 2, number_of_actuators))
    missing = {}
    for values, parameters in zip(
        (non_optimizable, optimizable), linear_actuator_parameters
    ):
        for index, key in parameters:
            column = torch.as_tensor(
                columns.get(
                    f"{prefix}{constants.actuator_parameters_key}/{key}",
                    np.full((number_of_rows, number_of_actuators), np.nan),
                )
            )
            missing[key] = column.isnan()
            values[:, index] = column.nan_to_num(0.0).float()

    # As in h5_scenario_parser.linear_actuators(): the first actuator rotates along the east axis, and its initial
    # angle is relative to the surface, which is stored pointing up, while ARTIST expects heliostats to initially
    # face south. The rotation from south to up, projected onto the east axis, is added to the initial angle.
    cpu = torch.device("cpu")
    axis, angle = rotations.rotation_angle_and_axis(
        from_orientation=torch.tensor([0.0, -1.0, 0.0, 0.0]),
        to_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        device=cpu,
    )
    optimizable[:, indices.actuator_initial_angle, indices.actuator_one_index] += (
        axis[indices.e] * angle
    )
    return non_optimizable, optimizable, missing


def convert_scenario_to_columnar(
    input_path: pathlib.Path | str, output_path: pathlib.Path | str
) -> None:
    """
    Convert a scenario file with heliostats in the legacy layout to the columnar layout.

    Everything apart from the heliostats is copied unchanged. Scenarios in the columnar layout load faster,
    especially for large heliostat fields, and yield exactly the same scenario.

    Parameters
    ----------
    input_path : pathlib.Path | str
        The scenario file with heliostats in the legacy layout.
    output_path : pathlib.Path | str
        The file to write the converted scenario to. It must differ from the input file.

    Raises
    ------
    ValueError
        If the heliostats of the input file are already stored in the columnar layout, or input and output are
        the same file.
    ColumnarLayoutError
        If the heliostats cannot be stored in the columnar layout.
    """
    if pathlib.Path(input_path).resolve() == pathlib.Path(output_path).resolve():
        raise ValueError("The converted scenario must be written to a different file.")
    with h5py.File(input_path, "r") as input_file:
        heliostats_config = input_file[constants.heliostat_key]
        if is_columnar(heliostats_config):
            raise ValueError(
                f"The heliostats in {input_path} are already stored in the columnar layout."
            )
        columns = heliostat_columns(read_legacy_heliostats(heliostats_config))
        with h5py.File(output_path, "w") as output_file:
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value
            for key in input_file.keys():
                if key != constants.heliostat_key:
                    input_file.copy(input_file[key], output_file, key)
            write_columnar_heliostats(
                output_file.create_group(constants.heliostat_key), columns
            )
