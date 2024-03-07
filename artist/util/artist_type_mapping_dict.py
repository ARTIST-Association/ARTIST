from artist.physics_objects.heliostats.alignment.kinematic.actuators.ideal_actuator import (
    IdealActuator,
)
from artist.physics_objects.heliostats.alignment.kinematic.rigid_body import (
    RigidBodyModule,
)
from artist.physics_objects.heliostats.concentrator.facets.point_cloud_facets import (
    PointCloudFacetModule,
)
from artist.util import config_dictionary

alignment_type_mapping = {config_dictionary.rigid_body_key: RigidBodyModule}

actuator_type_mapping = {config_dictionary.ideal_actuator_key: IdealActuator}

facet_type_mapping = {config_dictionary.point_cloud_facet_key: PointCloudFacetModule}
