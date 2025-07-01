import types

import pytest
import torch
from pytest_mock import MockerFixture

from artist.field.actuators import Actuators
from artist.field.kinematic import Kinematic


@pytest.mark.parametrize(
    "module",
    [
        Actuators,
        Kinematic,
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
