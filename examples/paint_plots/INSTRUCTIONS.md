# Example Code for ``PAINT`` Plots

This examples folder contains code to replicate the plots presented in our paper "PAINT: The First FAIR Database for
Concentrating Solar Power Plants".

## Configuration YAML

To make the execution of this code easier, the main configuration parameters are included in a ``paint_plot_config.yaml``
file. It is also possible to provide all these arguments are command line arguments when executing the scripts.
Additionally, if no arguments and no configuration file is provided, default values will be used -- which will probably
lead to the scripts failing.

Here is an overview of the configuration parameters and what they mean:

- `metadata_root`: The root directory in which the metadata will be saved, i.e. a folder with the name "metadata" will be saved within this directory.
- `metadata_file_name`: The file name for the metadata downloaded, if you do not change anything the STAC client from ``PAINT`` wil automatically download the metadata and save it to "calibration_metadata_all_heliostats.csv" in the "metadata" folder.
- `data_dir`: The directory in which all ``PAINT`` data will be saved. This data is required for the plots.
- `tower_file_name`: The name of the file containing the tower measurements. If you do not change anything, the STAC client from ``PAINT`` wil automatically download this data to the file "WRI1030197-tower-measurements.json" saved within the data directory.
- `scenarios_dir`: The name of the directory to save the ``ARTIST`` scenarios required for generating results.
- `results_dir`: The name of the directory to save the results from the calibration or flux prediction scenarios before plotting.
- `plots_dir`: The name of the directory to save the plots.
- `minimum_number_of_measurements`: The minimum number of calibration measurements required for a individual heliostats. Heliostats with less than this number will not be considered.
- `heliostats_for_raytracing`: A dictionary containing a mapping from a "heliostat ID" to a "calibration measurement ID". This is required for the flux prediction plot, where only certain heliostats are considered for ray tracing and a reference image is required for the plot.
- `device`: The device used for the computation.

## How to Use:

There are two plots present in our paper that are generated using ``ARTIST``. In order to replicate these plots it is important
to execute the code in the correct order.

### Run First

Regardless of which plot you wish to generate, you must first run the code to download the data, this consists of two scripts:

1. ``download_metadata.py``: This script will download all the metadata associated with the ``PAINT`` database. It will take a while to run.
2. ``download_data.py``: Based on the metadata, this script will now download all the required calibration, deflectometry, and tower data from the ``PAINT`` database required for the plots. It will also take a while to run.

### Calibration Plot

To replicate the calibration plots, please run the following scripts in the correct order:

1. TODO
2. TODO

### Flux Prediction Plot

To replicate the flux prediction plot, please run the following scripts in the correct order:

1. ``flux_prediction_scenario.py``: This script will generate the ``ARTIST`` scenario required for result calculation.
2. ``flux_prediction_raytracing.py``: This script will perform ray tracing in ``ARTIST`` and save the results ready for plotting.
3. ``flux_prediction_plot.py``: This script will generate the flux prediction plots and save them.
