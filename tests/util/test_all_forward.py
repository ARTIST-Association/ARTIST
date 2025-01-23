import types

import pytest
import torch
from pytest_mock import MockerFixture

from artist.field.actuator import Actuator
from artist.field.actuator_array import ActuatorArray
from artist.field.actuator_ideal import IdealActuator
from artist.field.actuator_linear import LinearActuator
from artist.field.facets_nurbs import NurbsFacet
from artist.field.heliostat import Heliostat
from artist.field.heliostat_field import HeliostatField
from artist.field.kinematic import Kinematic
from artist.field.kinematic_rigid_body import RigidBody
from artist.field.surface import Surface
from artist.field.tower_target_area import TargetArea
from artist.field.tower_target_area_array import TargetAreaArray
from artist.scene.light_source import LightSource
from artist.scene.light_source_array import LightSourceArray
from artist.scene.sun import Sun
from artist.util.nurbs import NURBSSurface


@pytest.mark.parametrize(
    "module",
    [
        ActuatorArray,
        LinearActuator,
        IdealActuator,
        NurbsFacet,
        HeliostatField,
        Heliostat,
        RigidBody,
        TargetAreaArray,
        TargetArea,
        Surface,
        LightSourceArray,
        Sun,
        NURBSSurface,
    ],
)
def test_forward_errors_of_subclasses(
    mocker: MockerFixture,
    module: torch.nn.Module,
) -> None:
    """
    Test the forward method of torch.nn.Module.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    module : torch.nn.Module
        A torch.nn.Module with a forward method.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    original_init = torch.nn.Module.__init__

    def mock_init(self):
        original_init(self)

    mocker.patch.object(module, "__init__", mock_init)

    module_instance = module()

    module_instance.forward = types.MethodType(module.forward, module_instance)

    with pytest.raises(NotImplementedError) as exc_info:
        module_instance()
    assert "Not Implemented!" in str(exc_info.value)


@pytest.mark.parametrize(
    "module",
    [
        Actuator,
        Kinematic,
        LightSource,
    ],
)
def test_forward_errors_of_base_classes(
    mocker: MockerFixture,
    module: torch.nn.Module,
) -> None:
    """
    Test the forward method of torch.nn.Module.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    module : torch.nn.Module
        A torch.nn.Module with a forward method.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    original_init = torch.nn.Module.__init__

    def mock_init(self):
        original_init(self)

    mocker.patch.object(module, "__init__", mock_init)

    module_instance = module()

    module_instance.forward = types.MethodType(module.forward, module_instance)

    with pytest.raises(NotImplementedError) as exc_info:
        module_instance()
    assert "Must be overridden!" in str(exc_info.value)
