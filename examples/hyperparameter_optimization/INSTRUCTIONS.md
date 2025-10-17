# Example Code for a Hyperparameter Optimization for the Surface Reconstruction

This examples folder contains code to replicate the hyperparameter optimization in our CVPR paper.

## Configuration YAML

To make the execution of this code easier, the main configuration parameters are included in a ``hpo_config.yaml``
file. It is also possible to provide all these arguments as command line arguments when executing the scripts.
Additionally, if no arguments and no configuration file is provided, default values will be used -- which will probably
lead to the scripts failing.

To make sure the configuration is successfully loaded, please provide the path to the configuration file via the ``--config``
command line argument when executing the script. If no argument is provided the script will look for the ``hpo_config.yaml``
located in the ``hyperparameter_optimization`` working directory, however this option is not failsafe, and we always suggest providing the command
line argument.

Here is an overview of the configuration parameters contained within the configuration file and what they mean:

- `metadata_root`: The root directory in which the metadata will be saved, i.e., a folder with the name "metadata" will be saved within this directory.
- `metadata_file_name`: The file name for the metadata downloaded, if you do not change anything the STAC client from ``PAINT`` will automatically download the metadata and save it to "calibration_metadata_all_heliostats.csv" in the "metadata" folder.
- `data_dir`: The directory in which all ``PAINT`` data will be saved. This data is required for the plots.
- `tower_file_name`: The name of the file containing the tower measurements. If you do not change anything, the STAC client from ``PAINT`` will automatically download this data to the file "WRI1030197-tower-measurements.json" saved within the data directory.
- `scenarios_dir`: The name of the directory to save the ``ARTIST`` scenarios required for generating results.
- `results_dir`: The name of the directory to save the results from the hyperparameter optimization and surface reconstruction before plotting.
- `plots_dir`: The name of the directory to save the plots.
- `propulate_logs_dir`: The name of the directory where the ``propulate``-hpo will save all logs.
- `heliostat_for_reconstruction`: A dictionary containing a mapping from a "heliostat ID" to a "calibration measurement ID". The provided heliostat will be reconstructed from the provided calibration measurements.
- `reconstruction_parameter_ranges`: The hyperparameter ranges handed to ``propulate`` from which to find optimal parameters.
- `device`: The device used for the computation.

## How to Use:

In order to replicate these hpo results presented in our paper it is important to execute the code in the correct order.

### Run First

You must first run the code to download the data, this consists of two scripts:

1. ``download_metadata.py``: This script will download all the metadata associated with the ``PAINT`` database. It will take a while to run.
2. ``download_data.py``: Based on the metadata, this script will now download all the required calibration, deflectometry, and tower data from the ``PAINT`` database required for the plots. It will also take a while to run.

### Hyperparameter Optimization with ``propulate``

The hyperparameter search is done using ``propulate``. Afterwards it is possible to visualize the results. Please run the following scripts in the correct order:

1. ``surface_reconstruction_viable_heliostats_list.py``: This script will iterate through the downloaded data and populate a list with file names that contain the measurements required for the hpo.
2. ``surface_reconstruction_generate_scenario.py``: This script will generate the ``ARTIST`` scenarios required for the hpo, the surface reconstruction and the plots.
3. ``surface_reconstruction_hyperparameter_search.py``: This script will perform the hpo with ``propulate`` and save the results for plotting.
4. ``surface_reconstruction_results.py``: This script will perform surface reconstruction once with the optimal hyperparameters and save the surface and flux results.
5. ``surface_reconstruction_plot.py``: This script will generate the flux prediction plots and reconstructed surface plots and save them.
