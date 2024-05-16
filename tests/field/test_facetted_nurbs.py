import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary
from artist.util.stral_converter import StralToSurfaceConverter

# Set variables in the script instead of using input function.
# Set this boolean to true.

# Set your values with the following variables.
STRAL_FILE_PATH = f"{ARTIST_ROOT}/tests/field/test_facetted_nurbs/stral_test_data"
HDF_FILE_PATH = "hi"
CONCENTRATOR_HEADER_NAME = "=5f2I2f"
FACET_HEADER_NAME = "=i9fI"
RAY_STRUCT_NAME = "=7f"
STEP_SIZE = 1000
TOLERANCE = 5e-4


@pytest.mark.parametrize(
    "extraction_method",
    [
        config_dictionary.convert_nurbs_from_points,
        config_dictionary.convert_nurbs_from_normals,
    ],
)
def test_facetted_nurbs(extraction_method: str) -> None:
    """
    Test the NURBS surface only, without raytracing.

    First a random surface is generated, it consists of ``surface_points``.
    Then, all the NURBS parameters are initialized (evaluation points, control points, degree,...)
    Next, the NURBS surface is initialized accordingly and then it is fitted to the
    random surface that was created in the beginning.
    The control points of the NURBS surface are the parameters of the optimizer.
    """
    torch.manual_seed(7)
    """Run the main function to start the conversion."""
    converter = StralToSurfaceConverter(
        stral_file_path=STRAL_FILE_PATH,
        surface_header_name=CONCENTRATOR_HEADER_NAME,
        facet_header_name=FACET_HEADER_NAME,
        points_on_facet_struct_name=RAY_STRUCT_NAME,
        step_size=STEP_SIZE,
    )
    converter.generate_surface_config_from_stral(
        extraction_method,
        number_eval_points_e=200,
        number_eval_points_n=200,
        number_control_points_e=10,
        number_control_points_n=10,
        tolerance=TOLERANCE,
    )

    # AssertionError(
    #     isinstance(facetted_nurbs, list)
    #     and all(isinstance(item, NURBSSurface) for item in facetted_nurbs),
    #     "The facetted NURBS is not a list of NURBS surfaces.",
    # )
