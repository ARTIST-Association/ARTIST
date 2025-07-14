from artist.field.actuators_ideal import IdealActuators
from artist.field.actuators_linear import LinearActuators
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.scene.sun import Sun
from artist.util import config_dictionary

heliostat_group_type_mapping = {
    f"{config_dictionary.rigid_body_key}_{config_dictionary.linear_actuator_key}": HeliostatGroupRigidBody,
    f"{config_dictionary.rigid_body_key}_{config_dictionary.ideal_actuator_key}": HeliostatGroupRigidBody,
}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct heliostat group type."""

actuator_type_mapping = {
    config_dictionary.linear_actuator_int: LinearActuators,
    config_dictionary.ideal_actuator_int: IdealActuators,
}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct actuator type."""

light_source_type_mapping = {config_dictionary.sun_key: Sun}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct light source type."""

nurbs_dimension_mapping = {
    config_dictionary.nurbs_u_dimension: 0,
    config_dictionary.nurbs_v_dimension: 1,
}