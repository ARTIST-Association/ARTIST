import pathlib

from artist import ARTIST_ROOT
from artist.util.configuration_classes import FacetConfig
from artist.util.surface_converter import SurfaceConverter


def test_surface_converter(device):
    """
    Test the surface converter with ``STRAL`` and ``PAINT`` files.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    heliostat_file_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/heliostat_properties.json"
    )
    deflectometry_file_path = pathlib.Path(ARTIST_ROOT) / "tests/data/deflectometry.h5"
    stral_file_path = pathlib.Path(ARTIST_ROOT) / "tests/data/stral_test_data.binp"

    surface_converter = SurfaceConverter(
        step_size=5000,
        number_eval_points_e=10,
        number_eval_points_n=10,
        number_control_points_e=5,
        number_control_points_n=5,
        max_epoch=1,
    )

    surface_config_paint = surface_converter.generate_surface_config_from_paint(
        heliostat_file_path=heliostat_file_path,
        deflectometry_file_path=deflectometry_file_path,
        device=device,
    )

    surface_config_stral = surface_converter.generate_surface_config_from_stral(
        stral_file_path=stral_file_path,
        device=device,
    )
    assert isinstance(surface_config_paint, list)
    assert isinstance(surface_config_stral, list)
    assert all(isinstance(obj, FacetConfig) for obj in surface_config_paint)
    assert all(isinstance(obj, FacetConfig) for obj in surface_config_stral)