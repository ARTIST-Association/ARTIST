import h5py


def print_attrs_to_file(name, obj, file):
    file.write(f"Path: {name}\n")

    # Print attributes if any
    for key, val in obj.attrs.items():
        file.write(f"  [Attribute] {key}: {val}\n")

    # Print dataset values if this is a dataset
    if isinstance(obj, h5py.Dataset):
        file.write(f"  [Dataset] Shape: {obj.shape}, Dtype: {obj.dtype}\n")
        try:
            data = obj[()]
            file.write(f"  [Value] {data}\n")
        except Exception as e:
            file.write(f"  [Error Reading Data] {e}\n")

    file.write("-" * 60 + "\n")


def save_hdf5_contents_to_txt(hdf5_filename, output_txt_filename):
    with h5py.File(hdf5_filename, "r") as f, open(output_txt_filename, "w") as out_file:
        f.visititems(lambda name, obj: print_attrs_to_file(name, obj, out_file))


# Example usage
if __name__ == "__main__":
    file_path = "/workVERLEIHNIX/mb/ARTIST/examples/field_optimizations/scenarios/deflectometry_scenario.h5"
    file_path2 = "/workVERLEIHNIX/mb/ARTIST/examples/field_optimizations/scenarios/ideal_baseline_scenario.h5"
    
    # output_txt_filename = "output.txt"  # Output file name
    # save_hdf5_contents_to_txt(file_path, output_txt_filename)

    # Open file in read/write mode
    # with h5py.File(file_path, "r+") as f:
    #     # Navigate to the dataset (example: "group1/dataset1")
    #     #del f["prototypes/actuator/actuator_1/type"]
    #     #f["prototypes/actuator/actuator_1"].create_dataset("type", data=b'linear')
        
    #     # del f["heliostats"]["AA28"]["actuator"]
    #     # del f["heliostats"]["AA28"]["kinematic"]
    #     # del f["heliostats"]["AA28"]["surface"]

    #     del f["heliostats/AA28/actuator/actuator_1/type"]
    #     f["heliostats/AA28/actuator/actuator_1"].create_dataset("type", data=b'linear')

    # print("Value updated successfully!")

    # import h5py

    with h5py.File(file_path2, 'r+') as src: 
        with h5py.File(file_path, 'w') as dst:
            src.copy('target_areas_cylindrical', dst)
            src.copy('target_areas_planar', dst)


