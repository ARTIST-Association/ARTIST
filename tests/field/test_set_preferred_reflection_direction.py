import pytest
from pytest_mock import MockerFixture

from artist.field.heliostat import Heliostat


def test_set_preferred_reflection_direction_error(
    mocker: MockerFixture,
) -> None:
    """
    Test the heliostat load from hdf5 method.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mock fixture used to create mock objects.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_heliostat = mocker.MagicMock(spec=Heliostat)
    mock_heliostat.is_aligned = False
    mock_rays = mocker.MagicMock()

    mock_heliostat.set_preferred_reflection_direction = Heliostat.set_preferred_reflection_direction.__get__(mock_heliostat, Heliostat)

    with pytest.raises(ValueError) as exc_info:
        mock_heliostat.set_preferred_reflection_direction(
            rays=mock_rays,
        )
    assert "Heliostat has not yet been aligned." in str(exc_info.value)
