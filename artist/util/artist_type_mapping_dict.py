"""This file implements a type mapping dictionary."""

from artist.field.actuator_ideal import (
    IdealActuator,
)
from artist.field.facets_point_cloud import (
    PointCloudFacet,
)
from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary

alignment_type_mapping = {config_dictionary.rigid_body_key: RigidBody}

actuator_type_mapping = {config_dictionary.ideal_actuator_key: IdealActuator}

facet_type_mapping = {config_dictionary.point_cloud_facet_key: PointCloudFacet}
