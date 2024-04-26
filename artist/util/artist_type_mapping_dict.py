"""This file implements a type mapping dictionary."""

from artist.field.actuator_ideal import (
    IdealActuator,
)
from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.scene import Sun
from artist.util import config_dictionary

light_source_type_mapping = {config_dictionary.sun_key: Sun}

kinematic_type_mapping = {config_dictionary.rigid_body_key: RigidBody}

actuator_type_mapping = {config_dictionary.ideal_actuator_key: IdealActuator}
