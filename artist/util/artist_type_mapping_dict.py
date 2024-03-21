"""This file implements a type mapping dictionary."""

from artist.physics_objects.actuator_ideal import (
    IdealActuator,
)
from artist.physics_objects.facets_point_cloud import (
    PointCloudFacetModule,
)
from artist.physics_objects.kinematic_rigid_body import (
    RigidBodyModule,
)
from artist.util import config_dictionary

alignment_type_mapping = {config_dictionary.rigid_body_key: RigidBodyModule}

actuator_type_mapping = {config_dictionary.ideal_actuator_key: IdealActuator}

facet_type_mapping = {config_dictionary.point_cloud_facet_key: PointCloudFacetModule}
