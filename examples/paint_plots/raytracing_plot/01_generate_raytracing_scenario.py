import logging
import pathlib
import sys
from pathlib import Path

import torch

from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary
from artist.util.environment_setup import get_device
from examples.paint_plots.helpers import join_safe, load_config

sys.path.append(str(Path(__file__).resolve().parent))
log = logging.getLogger(__name__)
"""A logger for the environment."""


def find_latest_deflectometry_file(name: str, paint_dir: str | Path) -> Path:
    """Find the latest deflectometry HDF5 file for a given heliostat.

    Parameters
    ----------
    name : str
        Heliostat name (e.g., "AA39").
    paint_dir : str | Path
        Base ``PAINT`` directory.

    Returns
    -------
    Path
        Path to the latest deflectometry file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    search_path = pathlib.Path(paint_dir) / name / "Deflectometry"
    pattern = f"{name}-filled*.h5"
    files = sorted(search_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No deflectometry file found for {name} in {search_path}"
        )
    return files[-1]


def generate_paint_scenario(
    paint_dir: str | Path,
    scenario_path: str | Path,
    tower_file: str | Path,
    heliostat_names: list[str],
    device: torch.device | str = "cpu",
    use_deflectometry: bool = True,
) -> None:
    """Generate an HDF5 scenario from ``PAINT`` inputs.

    Parameters
    ----------
    paint_dir : str | Path
        Base ``PAINT`` repository path.
    scenario_path : str | Path
        Output scenario path without extension ('.h5' will be added).
    tower_file : str | Path
        Path to the tower measurements HDF5.
    heliostat_names : list[str]
        Heliostat identifiers to include.
    device : torch.device | str, optional
        Torch device for processing, by default "cpu".
    use_deflectometry : bool, optional
        If True, include deflectometry data for surface fitting, by default True.
    """
    # Normalize to Path for mypy correctness
    scenario_path = pathlib.Path(scenario_path)
    tower_file_path = pathlib.Path(tower_file)

    if not scenario_path.parent.is_dir():
        raise FileNotFoundError(
            f"The folder ``{scenario_path.parent}`` selected to save the scenario does not exist. "
            "Please create the folder or adjust the file path before running again!"
        )

    # Prepare heliostat files.
    # The function is a hacky workaround to catch a MyPy error.

    def create_heliostat_files_list(
        names, paint_dir, config_dict, use_deflectometry=False
    ):
        if use_deflectometry:
            return [
                (
                    name,
                    pathlib.Path(
                        f"{paint_dir}/{name}/{config_dict.paint_properties_folder_name}/{name}{config_dict.paint_properties_file_name_ending}"
                    ),
                    find_latest_deflectometry_file(name, paint_dir),
                )
                for name in names
            ]
        else:
            return [
                (
                    name,
                    pathlib.Path(
                        f"{paint_dir}/{name}/{config_dict.paint_properties_folder_name}/{name}{config_dict.paint_properties_file_name_ending}"
                    ),
                )
                for name in names
            ]

    heliostat_files_list = create_heliostat_files_list(
        heliostat_names, paint_dir, config_dictionary, use_deflectometry
    )
    paths_arg = heliostat_files_list

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=tower_file_path,  # pass Path
            device=device,
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
    light_source_list_config = LightSourceListConfig(
        light_source_list=light_source_list
    )

    number_of_nurbs_control_points = torch.tensor([20, 20], device=device)
    nurbs_fit_method = config_dictionary.fit_nurbs_from_normals
    nurbs_deflectometry_step_size = 100
    nurbs_fit_tolerance = 1e-10
    nurbs_fit_max_epoch = 400

    # Please leave the optimizable parameters empty, they will automatically be added for the surface fit.
    nurbs_fit_optimizer = torch.optim.Adam(
        [torch.empty(1, requires_grad=True)], lr=1e-3
    )
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        paths=paths_arg,
        power_plant_position=power_plant_config.power_plant_position,
        number_of_nurbs_control_points=number_of_nurbs_control_points,
        deflectometry_step_size=nurbs_deflectometry_step_size,
        nurbs_fit_method=nurbs_fit_method,
        nurbs_fit_tolerance=nurbs_fit_tolerance,
        nurbs_fit_max_epoch=nurbs_fit_max_epoch,
        nurbs_fit_optimizer=nurbs_fit_optimizer,
        nurbs_fit_scheduler=nurbs_fit_scheduler,
        device=device,
    )

    # Generate the scenario given the defined parameters.
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,  # pass Path
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()


def main() -> None:
    config = load_config()

    device = torch.device(config["device"])
    device = get_device(device)

    # Map config -> args using new keys and base paths.
    paint_dir = pathlib.Path(config["paint_repository_base_path"])
    tower_file = join_safe(paint_dir, config["paint_tower_file"])

    examples_base = pathlib.Path(config["base_path"])
    heliostats = ["AA39", "AY26", "BC34"]

    scenario_base = join_safe(examples_base, config["raytracing_scenario_path"])

    # Generate two scenarios: deflectometry and ideal (no deflectometry).
    scenario_deflec_base = pathlib.Path(str(scenario_base) + "_deflectometry")
    scenario_ideal_base = pathlib.Path(str(scenario_base) + "_ideal")

    for base, use_def in [(scenario_deflec_base, True), (scenario_ideal_base, False)]:
        h5_path = base.with_suffix(".h5")
        if h5_path.exists():
            print(
                f"Scenario found at {h5_path}... continue without generating scenario."
            )
        else:
            print(
                f"Scenario not found. Generating a new one at {h5_path} (use_deflectometry={use_def})..."
            )
            generate_paint_scenario(
                paint_dir=str(paint_dir),
                scenario_path=base,
                tower_file=str(tower_file),
                heliostat_names=heliostats,
                device=device,
                use_deflectometry=use_def,
            )


if __name__ == "__main__":
    main()
