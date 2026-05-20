from artist.field.actuators_ideal import IdealActuators
from artist.field.actuators_linear import LinearActuators
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.scene.sun import Sun
from artist.util import constants

heliostat_group_type_mapping = {
    f"{constants.rigid_body_key}_{constants.linear_actuator_key}": HeliostatGroupRigidBody,
    f"{constants.rigid_body_key}_{constants.ideal_actuator_key}": HeliostatGroupRigidBody,
}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct heliostat group type."""

actuator_type_mapping = {
    constants.linear_actuator_int: LinearActuators,
    constants.ideal_actuator_int: IdealActuators,
}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct actuator type."""

light_source_type_mapping = {constants.sun_key: Sun}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct light source type."""
