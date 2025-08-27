import os
import pathlib
from typing import Optional, Union, Tuple
import json 
import h5py
from matplotlib import pyplot as plt
import numpy as np
import copy
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

def _generate_paint_scenario(
    scenario_path: Union[str, pathlib.Path],
    tower_file: Union[str, pathlib.Path],
    heliostat_files_list: list[Tuple[str, pathlib.Path]],
    device: str = "cpu",
) -> None:
    """Generate a scenario file based on the provided tower and heliostat data."""
    if not pathlib.Path(scenario_path).parent.is_dir():
        raise FileNotFoundError(
            f"The folder `{pathlib.Path(scenario_path).parent}` selected to save the scenario does not exist. "
            "Please create the folder or adjust the file path before running again!"
        )

    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=pathlib.Path(tower_file),  # cast to Path for mypy
            device=device
        )
    )

    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

    light_source_list = [light_source1_config]
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

    scenario_generator = ScenarioGenerator(
        file_path=pathlib.Path(scenario_path),  # cast to Path for mypy
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
    list[Tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    list[Tuple[str, pathlib.Path]]
]:
    """
    Load heliostat data from JSON and generate properties paths using the paint repository.
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

        properties_path = pathlib.Path(f"{paint_repo}/{name}/Properties/{name}-heliostat-properties.json")
        if properties_path.exists():
            heliostat_properties_list.append((name, properties_path))
        else:
            print(f"Warning: Missing properties file for {name} at {properties_path}")

    return heliostat_data_mapping, heliostat_properties_list

def load_or_create_scenario(
    scenario_path: Union[str, pathlib.Path],
    tower_file: Union[str, pathlib.Path],
    heliostat_properties_list: list[Tuple[str, pathlib.Path]],
    device: Optional[torch.device],
) -> Tuple[Scenario, pathlib.Path]:
    """Ensure scenario exists, create if missing, and return loaded Scenario and .h5 path."""
    scenario_path = pathlib.Path(scenario_path)
    if scenario_path.suffix != ".h5":
        scenario_path = scenario_path.with_suffix(".h5")

    if not scenario_path.exists():
        print(f"Scenario file not found at {scenario_path}. Generating scenario...")
        _generate_paint_scenario(
            scenario_path=scenario_path,
            tower_file=tower_file,
            heliostat_files_list=heliostat_properties_list,
            device=str(device) if device is not None else "cpu",
        )
    else:
        print(f"Scenario already exists at {scenario_path}. Skipping generation.")

    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_points_per_facet=torch.tensor([50, 50], device=device),
            device=device,
        )
    return scenario, scenario_path


def filter_valid_heliostat_data(
    heliostat_data_mapping: list[Tuple[str, list[pathlib.Path], list[pathlib.Path]]]
) -> list[Tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """Keep only flux images that have a matching calibration stem per heliostat."""
    valid = []
    for heliostat_name, valid_calibrations, flux_paths in heliostat_data_mapping:
        valid_stems = {p.stem.replace("-calibration-properties", "") for p in valid_calibrations}
        valid_flux_paths = [f for f in flux_paths if f.stem.replace("-flux", "") in valid_stems]
        valid.append((heliostat_name, valid_calibrations, valid_flux_paths))

    print("\nFiltered Heliostat Data Mapping:")
    for name, calibs, fluxes in valid:
        print(f"- {name}: {len(calibs)} valid calibrations, {len(fluxes)} matching flux images")
    return valid


def run_calibration(
    scenario_utis: Scenario,
    scenario_helios: Scenario,
    valid_heliostat_data_mapping: list[Tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    device: Optional[torch.device],
    centroids_extracted_by: Optional[list[str]] = None,
    # optimizer settings moved from inside function to parameters:
    use_ray_tracing: bool = False,
    tolerance: float = 0.05,
    max_epoch: int = 1000,
    initial_learning_rate: float = 0.001,
) -> dict:
    """
    Run distributed calibration for all groups and requested centroid sources.

    Parameters
    ----------
    use_ray_tracing : bool
        If True, disables motor_positions_calibration and expects ray-tracing-based optimization.
    tolerance : float
        Optimization tolerance (stopping/target).
    max_epoch : int
        Maximum number of optimization epochs.
    initial_learning_rate : float
        Learning rate for the optimizer.
    """
    centroids_extracted_by = centroids_extracted_by or [
        config_dictionary.paint_utis,
        config_dictionary.paint_helios,
    ]

    number_of_heliostat_groups = len(scenario_utis.heliostat_field.heliostat_groups)
    # Validate that UTIS and HeliOS scenarios have matching topology
    number_of_heliostat_groups_utis = len(scenario_utis.heliostat_field.heliostat_groups)
    number_of_heliostat_groups_helios = len(scenario_helios.heliostat_field.heliostat_groups)
    if number_of_heliostat_groups_utis != number_of_heliostat_groups_helios:
        raise ValueError(
            f"Mismatch in number of heliostat groups: UTIS={number_of_heliostat_groups_utis}, HeliOS={number_of_heliostat_groups_helios}. You have to load the same scenario twice!"
        )
    results_dict: dict = {}

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as (
        device_ctx,
        is_distributed,
        is_nested,
        rank,
        world_size,
        process_subgroup,
        groups_to_ranks_mapping,
        heliostat_group_rank,
        heliostat_group_world_size,
    ):
        # Use the device from context
        device_used = device_ctx

        for centroid in centroids_extracted_by:
            if centroid == config_dictionary.paint_utis:
                scenario = scenario_utis
            elif centroid == config_dictionary.paint_helios:
                scenario = scenario_helios
            else:
                raise ValueError(f"Unknown centroid source: {centroid}")
            for heliostat_group in scenario.heliostat_field.heliostat_groups:
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
                    device=device_used,
                )

                if heliostats_mask_calibration.sum() <= 0:
                    continue

                # Disable motor positions if using ray tracing mode
                if use_ray_tracing:
                    motor_positions_calibration = None

                optimizer = torch.optim.Adam(
                    [heliostat_group.kinematic.deviation_parameters.requires_grad_()],
                    lr=initial_learning_rate,
                )

                kinematic_optimizer = KinematicOptimizer(
                    scenario=scenario,
                    heliostat_group=heliostat_group,
                    optimizer=optimizer,
                )

                # Guard return value of optimize (may be None)
                raw_losses = kinematic_optimizer.optimize(
                    focal_spots_calibration=focal_spots_calibration,
                    incident_ray_directions=incident_ray_directions_calibration,
                    active_heliostats_mask=heliostats_mask_calibration,
                    target_area_mask_calibration=target_area_mask_calibration,
                    motor_positions_calibration=motor_positions_calibration,
                    tolerance=tolerance,
                    max_epoch=max_epoch,
                    num_log=max_epoch,
                    loss_type="l1",
                    loss_reduction="none",
                    loss_return_value="angular",
                    device=device_used,
                )
                if raw_losses is None:
                    continue
                losses = raw_losses  # expected torch.Tensor

                for index, name in enumerate(heliostat_group.names):
                    start = heliostats_mask_calibration[:index].sum()
                    end = heliostats_mask_calibration[: index + 1].sum()
                    per_heliostat_losses = losses[start:end].detach().cpu().numpy()
                    if name not in results_dict:
                        results_dict[name] = {}
                    results_dict[name][centroid] = per_heliostat_losses

    return results_dict


def attach_positions(results_dict: dict, scenario: Scenario) -> None:
    """Attach positions to results for all heliostats across all groups."""
    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            if name not in results_dict:
                results_dict[name] = {}
            results_dict[name]["position"] = position.clone().detach().cpu().tolist()

#----------------------
#public apis
#------------------

def plot_mrad_error_distributions(
    results_dict: dict,
    save_path: Optional[Union[str, pathlib.Path]] = None
) -> plt.Figure:
    """Plot histograms and KDEs of clipped mrad losses (0–11 mrad) for HeliOS and UTIS across all heliostats."""
    from scipy.stats import gaussian_kde

    helios_losses = []
    utis_losses = []

    for data in results_dict.values():
        helios_losses.extend(data[config_dictionary.paint_helios] * 1000)
        utis_losses.extend(data[config_dictionary.paint_utis] * 1000)

    x_vals = np.linspace(0, 10, 100)

    kde_helios = gaussian_kde(helios_losses, bw_method="scott")
    kde_utis = gaussian_kde(utis_losses, bw_method="scott")

    kde_vals_helios = kde_helios(x_vals)
    kde_vals_utis = kde_utis(x_vals)

    mean_helios = np.mean(helios_losses)
    mode_helios = x_vals[np.argmax(kde_vals_helios)]

    mean_utis = np.mean(utis_losses)
    mode_utis = x_vals[np.argmax(kde_vals_utis)]

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
        filename = f"{save_path}_mrad_distribution.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to {filename}")

    plt.close(fig)
    return fig


def plot_mrad_vs_distance(results_dict, save_path=None):
    """Plot mean pointing error (mrad) vs. heliostat XY distance with unique markers and trendlines.

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
    distances = []
    helios_means = []
    utis_means = []

    fig, ax = plt.subplots(figsize=FIGSIZE)
    

    for idx, (name, data) in enumerate(results_dict.items()):
        pos = np.array(data["position"])
        distance = np.linalg.norm(pos[:2])

        helios_mean = np.mean(data[config_dictionary.paint_helios])*1000 # results are calculated in rad but plotted in mrad.
        utis_mean = np.mean(data[config_dictionary.paint_utis])*1000 # results are calculated in rad but plotted in mrad.

        distances.append(distance)
        helios_means.append(helios_mean)
        utis_means.append(utis_mean)



        ax.scatter(
            distance,
            helios_mean,
            color=helmholtz_colors["hgfblue"],
            marker='o',
            label="HeliOS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )
        ax.scatter(
            distance,
            utis_mean,
            color=helmholtz_colors["hgfenergy"],
            marker='o',
            label="UTIS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )

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
    
    if save_path:
        filename = save_path+"_mrad_vs_distance.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved distance vs. error plot to {filename}")

    plt.close(fig)
    return fig


    #
def main(
    paint_repository: Union[str, pathlib.Path],
    heliostat_list_file: Union[str, pathlib.Path],
    scenario_path: Union[str, pathlib.Path],
    tower_file: Union[str, pathlib.Path],
    device: Optional[torch.device] = None,
    results_path: Union[str, pathlib.Path] = "calibration_results.pt",
    # Optimizer settings exposed here so they are easy to change:
    use_ray_tracing: bool = False,
    tolerance: float = 0.05,
    max_epoch: int = 1000,
    initial_learning_rate: float = 0.001,
) -> dict:
    """Run calibration pipeline, save results, and return results_dict."""
    heliostat_data_mapping, heliostat_properties_list = _load_heliostat_data(paint_repository, heliostat_list_file)

    scenario, scenario_path = load_or_create_scenario(
        scenario_path=scenario_path,
        tower_file=tower_file,
        heliostat_properties_list=heliostat_properties_list,
        device=device,
    )
    scenario_utis = copy.deepcopy(scenario)
    scenario_helios = copy.deepcopy(scenario)

    valid_heliostat_data_mapping = filter_valid_heliostat_data(heliostat_data_mapping)

    results_dict = run_calibration(
        scenario_utis=scenario_utis,
        scenario_helios=scenario_helios,
        valid_heliostat_data_mapping=valid_heliostat_data_mapping,
        device=device,
        centroids_extracted_by=[config_dictionary.paint_utis, config_dictionary.paint_helios],
        use_ray_tracing=use_ray_tracing,
        tolerance=tolerance,
        max_epoch=max_epoch,
        initial_learning_rate=initial_learning_rate,
    )

    attach_positions(results_dict, scenario)
    torch.save(results_dict, results_path)
    print(f"Calibration results saved to {results_path}")
    return results_dict


if __name__ == "__main__":
    # Inputs
    def load_config():
        """Load local example configuration from config.local.json."""
        script_dir = os.path.dirname(__file__) 
        candidates = [
            os.path.join(script_dir, "config.local.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        raise FileNotFoundError(
            "No config.local.json found. "
            "Copy config.example.json to config.local.json and customize it."
        )
    cfg = load_config()

    paint_base = pathlib.Path(cfg["paint_repository_base_path"])
    examples_base = pathlib.Path(cfg["examples_data_base_path"])

    def join_safe(base: pathlib.Path, maybe_rel: Union[str, pathlib.Path]) -> pathlib.Path:
        """Join base with maybe relative path, stripping leading separators."""
        s = str(maybe_rel)
        return base / s.lstrip("/\\")
    
    paint_repo = paint_base
    heliostat_list_file = join_safe(examples_base, cfg["examples_heliostat_list_file"])
    scenario_path = join_safe(examples_base, cfg["examples_scenario_path"])
    tower_file = join_safe(paint_base, cfg["paint_tower_file"])
    results_path = join_safe(examples_base, cfg["examples_results_path"])
    save_plot_path = cfg["examples_save_plot_path"]

    device = get_device()

    use_ray_tracing = cfg.get("01_examples_use_ray_tracing", False)
    tolerance = cfg.get("01_examples_tolerance", 0.035 if use_ray_tracing else 0.05)
    max_epoch = cfg.get("01_examples_max_epoch", 600 if use_ray_tracing else 1000)
    initial_learning_rate = cfg.get("01_examples_initial_learning_rate", 0.005 if use_ray_tracing else 0.003)

    # Run calibration.
    if results_path.exists():
        print(f"Found existing results at {results_path}. Skipping main().")
        results_dict = torch.load(results_path, weights_only=False)
    else:
        results_dict = main(
            paint_repository=paint_repo,
            heliostat_list_file=heliostat_list_file,
            scenario_path=scenario_path,
            tower_file=tower_file,
            device=device,
            results_path=results_path,
            use_ray_tracing=use_ray_tracing,
            tolerance=tolerance,
            max_epoch=max_epoch,
            initial_learning_rate=initial_learning_rate,
        )

    # Create plots.
    plot_mrad_vs_distance(results_dict, save_path=save_plot_path)
    plot_mrad_error_distributions(results_dict, save_path=save_plot_path)


