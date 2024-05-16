from artist.util.stral_converter import StralToSurfaceConverter

# Set variables in the script instead of using input function.
# Set this boolean to true.
USE_SCRIPT_VALUES = True
# Set your values with the following variables.
STRAL_FILE_PATH = "measurement_data\\stral_test_data"
HDF_FILE_PATH = "measurement_data\\stral_conversion_test"
CONCENTRATOR_HEADER_NAME = "=5f2I2f"
FACET_HEADER_NAME = "=i9fI"
RAY_STRUCT_NAME = "=7f"
STEP_SIZE = 2


def main():
    """Run the main function to start the conversion."""
    if USE_SCRIPT_VALUES:
        converter = StralToSurfaceConverter(
            stral_file_path=STRAL_FILE_PATH,
            hdf5_file_path=HDF_FILE_PATH,
            surface_header_name=CONCENTRATOR_HEADER_NAME,
            facet_header_name=FACET_HEADER_NAME,
            points_on_facet_struct_name=RAY_STRUCT_NAME,
            step_size=STEP_SIZE,
        )
    else:
        stral_file_path = str(
            input("Please enter the path to the STRAL file you want to convert:")
        )
        hdf5_file_path = str(
            input("Please enter the path to the hdf file you want to save:")
        )
        concentrator_header_name = str(
            input("Please enter the name of the concentrator header:")
        )
        facet_header_name = str(input("Please enter the name of the facet header:"))
        ray_struct_name = str(input("Please enter the name of the ray structure:"))
        step_size = int(input("Please enter the step size you want to use:"))
        converter = StralToSurfaceConverter(
            stral_file_path=stral_file_path,
            hdf5_file_path=hdf5_file_path,
            surface_header_name=concentrator_header_name,
            facet_header_name=facet_header_name,
            points_on_facet_struct_name=ray_struct_name,
            step_size=step_size,
        )

    converter.convert_stral_to_hdf5()


if __name__ == "__main__":
    main()
