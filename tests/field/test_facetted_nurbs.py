import pytest
import torch

from artist.field.nurbs import NURBSSurface
from artist.util.stral_converter import StralConverter

# Set variables in the script instead of using input function.
# Set this boolean to true.

# Set your values with the following variables.
STRAL_FILE_PATH = "tests\\field\\test_facetted_nurbs\\stral_test_data"
HDF_FILE_PATH = "tests\\field\\test_facetted_nurbs\\stral_conversion_test"
CONCENTRATOR_HEADER_NAME = "=5f2I2f"
FACET_HEADER_NAME = "=i9fI"
RAY_STRUCT_NAME = "=7f"
STEP_SIZE = 1000
TOLERANCE = 5e-4

pytest.mark.parametrize("extraction_method", ["deflectometry", "point cloud"])


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
    converter = StralConverter(
        stral_file_path=STRAL_FILE_PATH,
        hdf5_file_path=HDF_FILE_PATH,
        concentrator_header_name=CONCENTRATOR_HEADER_NAME,
        facet_header_name=FACET_HEADER_NAME,
        ray_struct_name=RAY_STRUCT_NAME,
        step_size=STEP_SIZE,
    )
    facetted_nurbs = converter.convert_stral_file_to_nurbs(
        extraction_method,
        num_control_points_e=10,
        num_control_points_n=10,
        tolerance=TOLERANCE,
    )

    AssertionError(
        isinstance(facetted_nurbs, list)
        and all(isinstance(item, NURBSSurface) for item in facetted_nurbs),
        "The facetted NURBS is not a list of NURBS surfaces.",
    )
