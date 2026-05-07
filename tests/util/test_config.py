from collections.abc import Callable

import pytest
import torch

from artist.util import constants
from artist.util.config import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    ActuatorPrototypeConfig,
    FacetConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicsConfig,
    KinematicsDeviations,
    KinematicsPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfaceConfig,
    SurfacePrototypeConfig,
    TargetAreaCylindricalConfig,
    TargetAreaCylindricalListConfig,
    TargetAreaPlanarConfig,
    TargetAreaPlanarListConfig,
)


@pytest.fixture
def tensor() -> torch.Tensor:
    """Return a simple 3-element float tensor for use in tests."""
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def normal() -> torch.Tensor:
    """Return a unit normal vector pointing in the up direction."""
    return torch.tensor([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "power_plant_config_dict, expected_keys",
    [
        (
            lambda power_plant_position: PowerPlantConfig(
                power_plant_position=power_plant_position
            ).create_power_plant_dict(),
            [constants.power_plant_position],
        ),
    ],
)
def test_power_plant_config(
    power_plant_config_dict: Callable[[torch.Tensor], dict[str, object]],
    expected_keys: list[str],
    tensor: torch.Tensor,
) -> None:
    """Test that ``PowerPlantConfig`` produces a dictionary with the expected keys."""
    dictionary = power_plant_config_dict(tensor)

    for key in expected_keys:
        assert key in dictionary


def test_target_area_planar(normal: torch.Tensor) -> None:
    """Test that ``TargetAreaPlanarConfig`` creates a dict with all required planar target area keys."""
    target_area_planar = TargetAreaPlanarConfig(
        target_area_key="area_1",
        center=torch.zeros(3),
        normal_vector=normal,
        plane_e=5.0,
        plane_u=10.0,
    )

    dictionary = target_area_planar.create_target_area_dict()

    assert constants.target_area_position_center in dictionary
    assert constants.target_area_normal_vector in dictionary
    assert constants.target_area_plane_e in dictionary
    assert constants.target_area_plane_u in dictionary


def test_target_area_planar_list(normal: torch.Tensor) -> None:
    """Test that ``TargetAreaPlanarListConfig`` creates a dict keyed by the target area name."""
    target_area_config = TargetAreaPlanarConfig(
        target_area_key="area_1",
        center=torch.zeros(3),
        normal_vector=normal,
        plane_e=1,
        plane_u=2,
    )

    list_config = TargetAreaPlanarListConfig(target_area_list=[target_area_config])
    dictionary = list_config.create_target_area_list_dict()

    assert "area_1" in dictionary


def test_target_area_cylindrical() -> None:
    """Test that ``TargetAreaCylindricalConfig`` creates a dict with all required cylindrical target area keys."""
    target_area_cylindrical = TargetAreaCylindricalConfig(
        target_area_key="cylinder_1",
        radius=3,
        center=torch.zeros(3),
        height=10,
        axis=torch.tensor([0, 0, 1]),
        normal=torch.tensor([1, 0, 0]),
        opening_angle=0.5,
    )

    dictionary = target_area_cylindrical.create_target_area_dict()

    assert constants.target_area_cylinder_radius in dictionary
    assert constants.target_area_cylinder_center in dictionary
    assert constants.target_area_cylinder_height in dictionary


def test_target_area_cylindrical_list() -> None:
    """Test that ``TargetAreaCylindricalListConfig`` creates a dict keyed by the target area name."""
    target_area = TargetAreaCylindricalConfig(
        target_area_key="cylindrical",
        radius=1,
        center=torch.zeros(3),
        height=2,
        axis=torch.tensor([0, 0, 1]),
        normal=torch.tensor([1, 0, 0]),
        opening_angle=0.5,
    )

    list_config = TargetAreaCylindricalListConfig(target_area_list=[target_area])
    dictionary = list_config.create_target_area_list_dict()

    assert "cylindrical" in dictionary


@pytest.fixture
def light_source() -> LightSourceConfig:
    """Return a ``LightSourceConfig`` instance configured with a normal distribution."""
    return LightSourceConfig(
        light_source_key="sun",
        light_source_type="sun",
        number_of_rays=100,
        distribution_type=constants.light_source_distribution_is_normal,
        mean=0.0,
        covariance=1.0,
    )


def test_light_source_dict(light_source: LightSourceConfig) -> None:
    """Test that ``LightSourceConfig`` produces a dict with type, ray count, and distribution keys."""
    dictionary = light_source.create_light_source_dict()

    assert constants.light_source_type in dictionary
    assert constants.light_source_number_of_rays in dictionary
    assert constants.light_source_distribution_parameters in dictionary


def test_light_source_list(light_source: LightSourceConfig) -> None:
    """Test that ``LightSourceListConfig`` creates a dict keyed by the light source name."""
    list_config = LightSourceListConfig(light_source_list=[light_source])
    dictionary = list_config.create_light_source_list_dict()

    assert "sun" in dictionary


def test_light_source_invalid_distribution() -> None:
    """Test that ``LightSourceConfig`` raises ValueError for an unknown distribution type."""
    with pytest.raises(ValueError) as exc_info:
        LightSourceConfig(
            light_source_key="sun",
            light_source_type="sun",
            number_of_rays=100,
            distribution_type="invalid",
            mean=0.0,
            covariance=1.0,
        )

    assert "Unknown light source distribution type" in str(exc_info.value)


@pytest.fixture
def facet() -> FacetConfig:
    """Return a ``FacetConfig`` instance with minimal valid parameters."""
    return FacetConfig(
        facet_key="facet_1",
        control_points=torch.zeros((3, 3)),
        degrees=torch.tensor([2, 2]),
        translation_vector=torch.zeros(3),
        canting=torch.zeros(2),
    )


def test_facet_dict(facet: FacetConfig) -> None:
    """Test that ``FacetConfig`` produces a dict with control point and degree keys."""
    dictionary = facet.create_facet_dict()

    assert constants.facet_control_points in dictionary
    assert constants.facet_degrees in dictionary


@pytest.mark.parametrize(
    "surface_class",
    [
        SurfaceConfig,
        SurfacePrototypeConfig,
    ],
)
def test_surface_configs(surface_class: type, facet: FacetConfig) -> None:
    """Test that ``SurfaceConfig`` and ``SurfacePrototypeConfig`` produce dicts with the facets key."""
    surface_config = surface_class(facet_list=[facet])
    dictionary = surface_config.create_surface_dict()

    assert constants.facets_key in dictionary
    assert "facet_1" in dictionary[constants.facets_key]


@pytest.fixture
def kinematics_deviations() -> KinematicsDeviations:
    """Return a ``KinematicsDeviations`` instance with all deviation parameters set."""
    return KinematicsDeviations(
        first_joint_translation_e=torch.tensor(1.0),
        first_joint_translation_n=torch.tensor(2.0),
        first_joint_translation_u=torch.tensor(3.0),
        first_joint_tilt_n=torch.tensor(4.0),
        first_joint_tilt_u=torch.tensor(5.0),
        second_joint_translation_e=torch.tensor(6.0),
        second_joint_translation_n=torch.tensor(7.0),
        second_joint_translation_u=torch.tensor(8.0),
        second_joint_tilt_e=torch.tensor(9.0),
        second_joint_tilt_n=torch.tensor(10.0),
        concentrator_translation_e=torch.tensor(11.0),
        concentrator_translation_n=torch.tensor(12.0),
        concentrator_translation_u=torch.tensor(13.0),
    )


@pytest.mark.parametrize(
    "kinematics_config",
    [
        KinematicsConfig,
        KinematicsPrototypeConfig,
    ],
)
def test_kinematics_configs(
    kinematics_config: type, kinematics_deviations: KinematicsDeviations
) -> None:
    """Test that ``KinematicsConfig`` and ``KinematicsPrototypeConfig`` produce dicts with the required keys."""
    kinematics = kinematics_config(
        kinematics_type="test",
        initial_orientation=torch.tensor([0, 0, 1]),
        deviations=kinematics_deviations,
    )

    dictionary = kinematics.create_kinematics_dict()

    assert constants.kinematics_type in dictionary
    assert constants.kinematics_initial_orientation in dictionary
    assert constants.kinematics_deviations in dictionary


def test_actuator_parameters() -> None:
    """Test that ``ActuatorParameters`` produces a dict with all required actuator parameter keys."""
    actuator_parameters = ActuatorParameters(
        increment=torch.tensor(1.0),
        initial_stroke_length=torch.tensor(2.0),
        offset=torch.tensor(3.0),
        pivot_radius=torch.tensor(4.0),
        initial_angle=torch.tensor(5.0),
    )

    dictionary = actuator_parameters.create_actuator_parameters_dict()

    assert constants.actuator_increment in dictionary
    assert constants.actuator_initial_stroke_length in dictionary
    assert constants.actuator_offset in dictionary
    assert constants.actuator_pivot_radius in dictionary
    assert constants.actuator_initial_angle in dictionary


@pytest.fixture
def actuator() -> ActuatorConfig:
    """Return an ``ActuatorConfig`` instance with minimal valid parameters."""
    return ActuatorConfig(
        key="actuator_1",
        actuator_type="linear",
        clockwise_axis_movement=True,
        min_max_motor_positions=[0, 1],
        parameters=ActuatorParameters(
            increment=torch.tensor(1.0),
            initial_stroke_length=torch.tensor(2.0),
        ),
    )


def test_actuator_config(actuator: ActuatorConfig) -> None:
    """Test that ``ActuatorConfig`` produces a dict with the actuator type and motor position keys."""
    dictionary = actuator.create_actuator_dict()

    assert constants.actuator_type_key in dictionary
    assert constants.actuator_min_max_motor_positions in dictionary


@pytest.mark.parametrize(
    "list_class",
    [
        ActuatorListConfig,
        ActuatorPrototypeConfig,
    ],
)
def test_actuator_lists(list_class: type, actuator: ActuatorConfig) -> None:
    """Test that ``ActuatorListConfig`` and ``ActuatorPrototypeConfig`` create dicts keyed by actuator name."""
    list_config = list_class(actuator_list=[actuator])
    dictionary = list_config.create_actuator_list_dict()

    assert "actuator_1" in dictionary


def test_prototype_config(facet: FacetConfig, actuator: ActuatorConfig) -> None:
    """Test that ``PrototypeConfig`` produces a dict with surface, kinematics, and actuator prototype keys."""
    surface_config = SurfacePrototypeConfig(facet_list=[facet])

    kinematics_config = KinematicsPrototypeConfig(
        kinematics_type="test",
        initial_orientation=torch.tensor([0, 0, 1]),
    )

    actuator_config = ActuatorPrototypeConfig(actuator_list=[actuator])

    prototype = PrototypeConfig(surface_config, kinematics_config, actuator_config)

    dictionary = prototype.create_prototype_dict()

    assert constants.surface_prototype_key in dictionary
    assert constants.kinematics_prototype_key in dictionary
    assert constants.actuators_prototype_key in dictionary


def test_heliostat_all_branches(
    facet: FacetConfig,
    kinematics_deviations: KinematicsDeviations,
    actuator: ActuatorConfig,
) -> None:
    """Test that a fully configured ``HeliostatConfig`` produces a dict with surface, kinematics, and actuator keys."""
    surface = SurfaceConfig(facet_list=[facet])

    kinematics = KinematicsConfig(
        kinematics_type="test",
        initial_orientation=torch.tensor([0, 0, 1]),
        deviations=kinematics_deviations,
    )

    actuators = ActuatorListConfig(actuator_list=[actuator])

    heliostat = HeliostatConfig(
        name="h_full",
        heliostat_id=10,
        position=torch.zeros(3),
        surface=surface,
        kinematics=kinematics,
        actuators=actuators,
    )

    d = heliostat.create_heliostat_config_dict()

    assert constants.heliostat_surface_key in d
    assert constants.heliostat_kinematics_key in d
    assert constants.heliostat_actuator_key in d


@pytest.fixture
def heliostat() -> HeliostatConfig:
    """Return a minimal ``HeliostatConfig`` instance with name, id, and position."""
    return HeliostatConfig(
        name="heliostat_1",
        heliostat_id=1,
        position=torch.zeros(3),
    )


def test_heliostat_list(heliostat: HeliostatConfig) -> None:
    """Test that ``HeliostatListConfig`` creates a dict keyed by the heliostat name."""
    list_config = HeliostatListConfig(heliostat_list=[heliostat])
    dictionary = list_config.create_heliostat_list_dict()

    assert "heliostat_1" in dictionary
