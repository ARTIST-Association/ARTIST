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
    with h5py.File(hdf5_filename, 'r') as f, open(output_txt_filename, 'w') as out_file:
        f.visititems(lambda name, obj: print_attrs_to_file(name, obj, out_file))

# Example usage
if __name__ == "__main__":
    hdf5_filename = "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups_ideal.h5"         # Replace with your HDF5 file
    output_txt_filename = "output.txt"     # Output file name
    save_hdf5_contents_to_txt(hdf5_filename, output_txt_filename)
