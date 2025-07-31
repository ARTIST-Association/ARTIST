import pathlib
from typing import Optional, Union
import h5py
from matplotlib import pyplot as plt
import numpy as np
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import LightSourceConfig, LightSourceListConfig
from artist.scenario.scenario import Scenario
from artist.scenario.scenario_generator import ScenarioGenerator
from artist.util import config_dictionary
from artist.util import set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

FIGSIZE = (6,4)
LEGEND_FONTSIZE = 8

helmholtz_colors = {
    "hgfblue": "#005AA0",
    "hgfdarkblue": "#0A2D6E",
    "hgfgreen": "#8CB423",
    "hgfgray": "#5A696E",
    "hgfaerospace": "#50C8AA",
    "hgfearthandenvironment": "#326469",
    "hgfenergy": "#FFD228",
    "hgfhealth": "#D23264",
    "hgfkeytechnologies": "#A0235A",
    "hgfmatter": "#F0781E",
}
# Set up logger
set_logger_config()



#---------------
#helper functions
#---------------

import json
import pathlib
from typing import List, Tuple


def _generate_paint_scenario(scenario_path, tower_file, heliostat_files_list, device="cpu"):

    if not pathlib.Path(scenario_path).parent.is_dir():
        raise FileNotFoundError(
            f"The folder ``{pathlib.Path(scenario_path).parent}`` selected to save the scenario does not exist. "
            "Please create the folder or adjust the file path before running again!"
        )

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)

    target_area = [
        target_area
        for target_area in target_area_list_config.target_area_list
        if target_area.target_area_key == config_dictionary.target_area_receiver
    ]

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        aim_point=target_area[0].center,
        device=device,
    )
    """Generate the scenario given the defined parameters."""
    scenario_generator = ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()

def _load_heliostat_data(
    paint_repo: Union[str, pathlib.Path],
    input_path: Union[str, pathlib.Path]
) -> Tuple[
    List[Tuple[str, List[pathlib.Path], List[pathlib.Path]]],
    List[Tuple[str, pathlib.Path]]
]:
    """
    Load heliostat data from JSON and generate properties paths using paint_repo.

    Parameters
    ----------
    paint_repo : str or Path
        Base directory of the PAINT repo (e.g., /workVERLEIHNIX/share/PAINT)
    input_path : str or Path
        Path to the JSON file.

    Returns
    -------
    Tuple of:
    - List of (name, [calibrations], [flux_images])
    - List of (name, heliostat-properties.json path), only if the file exists
    """
    input_path = pathlib.Path(input_path).resolve()
    paint_repo = pathlib.Path(paint_repo).resolve()

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    heliostat_data_mapping = []
    heliostat_properties_list = []

    for item in raw_data:
        name = item["name"]
        calibrations = [pathlib.Path(p) for p in item["calibrations"]]
        flux_images = [pathlib.Path(p) for p in item["flux_images"]]

        heliostat_data_mapping.append((name, calibrations, flux_images))

        # Construct and check for the properties file
        properties_path = pathlib.Path(f"{paint_repo}/{name}/Properties/{name}-heliostat-properties.json")
        if properties_path.exists():
            heliostat_properties_list.append((name, properties_path))
        else:
            print(f"Warning: Missing properties file for {name} at {properties_path}")

    return heliostat_data_mapping, heliostat_properties_list





#----------------------
#public apis
#------------------

def plot_mrad_error_distributions(results_dict, save_path=None):
    """
    Plots histograms + KDEs of clipped mrad losses (0–11 mrad) for HeliOS and UTIS across all heliostats.

    Parameters
    ----------
    results_dict : dict
        Dictionary with structure: {
            heliostat_name: {
                "mrad_losses_helios": [...],
                "mrad_losses_utis": [...],
                ...
            }
        }
    save_path : Path or str, optional
        If given, saves the plot to this path.
    """
    from scipy.stats import gaussian_kde

    # Gather and process all losses
    helios_losses = []
    utis_losses = []

    for data in results_dict.values():
        helios_losses.extend(data[config_dictionary.paint_helios]*1000) # multiplication because results are calculated in rad but plotted in mrad
        utis_losses.extend(data[config_dictionary.paint_utis]*1000)
        
    x_vals = np.linspace(0, 10, 100)

    # KDEs
    kde_helios = gaussian_kde(helios_losses, bw_method="scott")
    kde_utis = gaussian_kde(utis_losses, bw_method="scott")

    kde_vals_helios = kde_helios(x_vals)
    kde_vals_utis = kde_utis(x_vals)

    mean_helios = np.mean(helios_losses)
    mode_helios = x_vals[np.argmax(kde_vals_helios)]

    mean_utis = np.mean(utis_losses)
    mode_utis = x_vals[np.argmax(kde_vals_utis)]
    
    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(helios_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="HeliOS Histogram", color=helmholtz_colors["hgfblue"])
    ax.plot(x_vals, kde_vals_helios, label="HeliOS KDE", color=helmholtz_colors["hgfblue"])
    ax.axvline(mode_helios, color=helmholtz_colors["hgfblue"], linestyle="--", label=f"HeliOS Mode: {mode_helios:.2f} mrad")
    ax.axvline(mean_helios, color=helmholtz_colors["hgfblue"], linestyle=":", label=f"HeliOS Mean: {mean_helios:.2f} mrad")

    ax.hist(utis_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="UTIS Histogram", color=helmholtz_colors["hgfenergy"])
    ax.plot(x_vals, kde_vals_utis, label="UTIS KDE", color=helmholtz_colors["hgfenergy"])
    ax.axvline(mode_utis, color=helmholtz_colors["hgfenergy"], linestyle="--", label=f"UTIS Mode: {mode_utis:.2f} mrad")
    ax.axvline(mean_utis, color=helmholtz_colors["hgfenergy"], linestyle=":", label=f"UTIS Mean: {mean_utis:.2f} mrad")

    ax.set_xlabel("Pointing Error / mrad")
    ax.set_ylabel("Density / -")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True)

    if save_path:
        filename = save_path+"_mrad_distribution.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to {save_path}")

    plt.close(fig)
    return fig


def plot_mrad_vs_distance(results_dict, save_path=None):
    """
    Plots mean pointing error (mrad) vs. heliostat distance in XY plane from origin (0, 0),
    with unique markers per heliostat and trendlines.

    Parameters
    ----------
    results_dict : dict
        Dictionary with structure:
            {
                heliostat_name: {
                    "mrad_losses_helios": [...],
                    "mrad_losses_utis": [...],
                    "position": [x, y, z]
                },
                ...
            }
    save_path : str or Path, optional
        If given, saves the plot to this path.
    """
    # Marker cycle (repeatable)
    marker_styles = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'H', '>', '<']
    
    distances = []
    helios_means = []
    utis_means = []

    fig, ax = plt.subplots(figsize=FIGSIZE)
    

    for idx, (name, data) in enumerate(results_dict.items()):
        pos = np.array(data["position"])
        distance = np.linalg.norm(pos[:2])

        helios_mean = np.mean(data[config_dictionary.paint_helios])*1000 # results are calculated in rad but plotted in mrad
        utis_mean = np.mean(data[config_dictionary.paint_utis])*1000 # results are calculated in rad but plotted in mrad

        distances.append(distance)
        helios_means.append(helios_mean)
        utis_means.append(utis_mean)

        marker = marker_styles[idx % len(marker_styles)]

        ax.scatter(
            distance,
            helios_mean,
            color=helmholtz_colors["hgfblue"],
            marker=marker,
            label="HeliOS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )
        ax.scatter(
            distance,
            utis_mean,
            color=helmholtz_colors["hgfenergy"],
            marker=marker,
            label="UTIS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )

    # Convert to arrays for fitting
    distances = np.array(distances)
    helios_means = np.array(helios_means)
    utis_means = np.array(utis_means)

    # Trendlines
    helios_fit = np.poly1d(np.polyfit(distances, helios_means, 1))
    utis_fit = np.poly1d(np.polyfit(distances, utis_means, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax.plot(x_vals, helios_fit(x_vals), color=helmholtz_colors["hgfblue"], linestyle="--", label="HeliOS Trend")
    ax.plot(x_vals, utis_fit(x_vals), color=helmholtz_colors["hgfenergy"], linestyle="--", label="UTIS Trend")

    ax.set_xlabel("Heliostat Distance to Tower / m")
    ax.set_ylabel("Mean Pointing Error / mrad")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", ncol=2)
    #ax.set_ylim([0,16])
    
    if save_path:
        filename = save_path+"_mrad_vs_distance.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved distance vs. error plot to {filename}")

    plt.close(fig)
    return fig


    #
def main():
    #Step 0: load heliostat list from pregenerated file
    heliostat_data_mapping, heliostat_properties_list = _load_heliostat_data(paint_repo,heliostat_list_file)


    # Step 1: Generate scenario only if it doesn't exist
    scenario_file = scenario_path
    if not pathlib.Path(scenario_file).exists():
        print(f"Scenario file not found at {scenario_path}. Generating scenario...")
        _generate_paint_scenario(
            scenario_path=scenario_path,
            tower_file=tower_file,
            heliostat_files_list=heliostat_properties_list,
            device=device
        )
    else:
        print(f"Scenario already exists at {scenario_path}. Skipping generation.")


    # Load the scenario.

    if scenario_path.suffix != '.h5':
        scenario_path = scenario_path.with_suffix('.h5')

    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )


    number_of_heliostat_groups = scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as (
        device,
        is_distributed,
        is_nested,
        rank,
        world_size,
        process_subgroup,
        groups_to_ranks_mapping,
        heliostat_group_rank,
        heliostat_group_world_size,
    ):
        # Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
        # Please follow the following style: list[tuple[str, list[pathlib.Path]]]
        

        # Filtered those files which havoutput list
        valid_heliostat_data_mapping =[]

        for heliostat_name, valid_calibrations, flux_paths in heliostat_data_mapping:
            # Include only the flux paths that match a valid calibration stem
            valid_stems = {p.stem.replace("-calibration-properties", "") for p in valid_calibrations}
            valid_flux_paths = [
                f for f in flux_paths
                if f.stem.replace("-flux", "") in valid_stems
            ]
            valid_heliostat_data_mapping.append((heliostat_name, valid_calibrations, valid_flux_paths))

        print("\nFiltered Heliostat Data Mapping:")
        for name, calibs, fluxes in valid_heliostat_data_mapping:
            print(f"- {name}: {len(calibs)} valid calibrations, {len(fluxes)} matching flux images")

        centroids_extracted_by = [ config_dictionary.paint_utis, config_dictionary.paint_helios]
        results_path = pathlib.Path("calibration_results.pt")  # or set a full path as needed

        results_dict = {}
        for centroid in centroids_extracted_by:
            for heliostat_group_index, heliostat_group in enumerate(
                scenario.heliostat_field.heliostat_groups
            ):
                # Load the calibration data.
                (
                    focal_spots_calibration,
                    incident_ray_directions_calibration,
                    motor_positions_calibration,
                    heliostats_mask_calibration,
                    target_area_mask_calibration,
                ) = paint_loader.extract_paint_calibration_properties_data(
                    heliostat_calibration_mapping=[
                        (heliostat_name, calibration_properties_paths)
                        for heliostat_name, calibration_properties_paths, _ in valid_heliostat_data_mapping
                        if heliostat_name in heliostat_group.names
                    ],
                    heliostat_names=heliostat_group.names,
                    target_area_names=scenario.target_areas.names,
                    power_plant_position=scenario.power_plant_position,
                    centroid_extrected_by=centroid,
                    device=device,
                )
                if heliostats_mask_calibration.sum() > 0:
                    # Set up optimizer and scheduler.
                    tolerance = 0.0005
                    max_epoch = 10000
                    initial_learning_rate = 0.00005

                    use_ray_tracing = False
                    if use_ray_tracing:
                        motor_positions_calibration = None
                        tolerance = 0.035
                        max_epoch = 600
                        initial_learning_rate = 0.0004

                    optimizer = torch.optim.Adam(
                        [heliostat_group.kinematic.deviation_parameters.requires_grad_()], lr=initial_learning_rate #heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_()
                    )

                    # Create the kinematic optimizer.
                    kinematic_optimizer = KinematicOptimizer(
                        scenario=scenario,
                        heliostat_group=heliostat_group,
                        optimizer=optimizer,
                    )

                    # Calibrate the kinematic.
                    
                    losses = kinematic_optimizer.optimize(
                        focal_spots_calibration=focal_spots_calibration,
                        incident_ray_directions=incident_ray_directions_calibration,
                        active_heliostats_mask=heliostats_mask_calibration,
                        target_area_mask_calibration=target_area_mask_calibration,
                        motor_positions_calibration=motor_positions_calibration,
                        tolerance=tolerance,
                        max_epoch=max_epoch,
                        num_log=max_epoch,
                        loss_type = "l1",
                        loss_reduction= "none",
                        loss_return_value="angular",
                        device=device
                    )

                    losses_per_heliostat = []

                    for index, name in enumerate(scenario.heliostat_field.heliostat_groups[0].names):
                        start = heliostats_mask_calibration[:index].sum()
                        end = heliostats_mask_calibration[:index + 1].sum()
                        losses_per_heliostat.append(losses[start:end].detach().cpu().numpy())
                        
                        if name not in results_dict:
                                results_dict[name] = {}  # Create entry if it doesn't exist

                            # Initialize the results dictionary
                        results_dict[name][centroid] = losses_per_heliostat[index]



    # Define the results output path


    heliostat_names = scenario.heliostat_field.heliostat_groups[0].names
    heliostat_positions = scenario.heliostat_field.heliostat_groups[0].positions

    for name, position in zip(heliostat_names, heliostat_positions):
        if name not in results_dict:
            results_dict[name] = {}

        results_dict[name]["position"] = position.clone().detach().cpu().tolist()


    # Save to disk
    torch.save(results_dict, results_path)


# Set the device
device = get_device()
# Specify the path to your scenario.h5 file.
heliostat_list_file = "examples/data/heliostat_files.json"
paint_repo = "/workVERLEIHNIX/share/PAINT"
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mp/ARTIST/examples/data/scenarios/heliostat_calibration_paint"
)
# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path(
    "/workVERLEIHNIX/mp/ARTIST/examples/data/tower-measurements.json"
)

save_plot_path = "/workVERLEIHNIX/mp/ARTIST/examples/plots/01_paint_raytracing_example_plot"

main() #TODO hier morgen weiter machen
print(f"Calibration results saved to {results_path}")
plot_mrad_vs_distance(results_dict, save_path=save_plot_path )
plot_mrad_error_distributions(results_dict, save_path=save_plot_path)
