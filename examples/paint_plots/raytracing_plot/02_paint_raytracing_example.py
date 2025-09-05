import pathlib
from typing import Dict

import h5py
import numpy as np
import torch
from PIL import Image

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.data_loader.paint_loader import (
    extract_canting_and_translation_from_properties,
)
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
from artist.util.environment_setup import get_device
from examples.paint_plots.helpers import (
    join_safe,
    load_config,
    perform_inverse_canting_and_translation,
)

MEASUREMENT_IDS = {"AA39": 149576, "AY26": 247613, "BC34": 82084}


def load_image_as_tensor(
    name: str,
    paint_dir: str | pathlib.Path,
    measurement_id: int,
    image_key: str,
) -> torch.Tensor:
    """Load a flux PNG as grayscale and return a float tensor in [0, 1].

    Parameters
    ----------
    name : str
        Heliostat name.
    paint_dir : str | Path
        Base ''PAINT'' directory.
    measurement_id : int
        Measurement identifier.
    image_key : str
        Image key suffix (e.g., "cropped", "flux").

    Returns
    -------
    torch.Tensor
        Grayscale image tensor with shape (H, W), dtype float32 in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    """
    # Build the path.
    image_path = pathlib.Path(
        f"{paint_dir}/{name}/{config_dictionary.paint_calibration_folder_name}/{measurement_id}-{image_key}.png"
    )

    # Open image in grayscale ('L' mode).
    image = Image.open(image_path).convert("L")

    # Convert to tensor and normalize to [0, 1] float.
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return tensor


def align_and_trace_rays(
    scenario: Scenario,
    aim_points: torch.Tensor,
    light_direction: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    target_area_mask: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Align heliostats and perform heliostat ray tracing.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario object.
    aim_points : torch.Tensor
        Aim points on the receiver for each heliostat; shape compatible with the
        scenario's alignment routine.
    light_direction : torch.Tensor
        Incoming light directions per heliostat; shape compatible with the scenario.
    active_heliostats_mask : torch.Tensor
        Mask indicating which heliostats are active.
    target_area_mask : torch.Tensor
        Target area indices for each active heliostat.
    device : torch.device | str, optional
        Device to use for computations, by default "cuda".

    Returns
    -------
    torch.Tensor
        Flux density image(s) on the receiver.
    """
    # Activate heliostats.
    scenario.heliostat_field.heliostat_groups[0].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask
    )

    # Align all heliostats.
    scenario.heliostat_field.heliostat_groups[
        0
    ].align_surfaces_with_incident_ray_directions(
        aim_points=aim_points,
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=scenario.heliostat_field.heliostat_groups[0]
    )

    # Perform heliostat-based ray tracing.
    return ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )


def generate_flux_images(
    scenario_path: str | pathlib.Path,
    heliostats: list[str],
    paint_dir: str | pathlib.Path,
    result_file: str | pathlib.Path,
    device: torch.device | str,
    result_key: str = "flux_deflectometry",
) -> None:
    """Generate flux images via alignment and ray tracing and save to a merged result file.

    Parameters
    ----------
    scenario_path : str | Path
        Base path (without .h5) of the scenario to load.
    heliostats : list[str]
        Heliostat names to process, order must match extracted calibration data.
    paint_dir : str | Path
        ''PAINT'' repository path.
    result_file : str | Path
        Path to a single merged .pt file to store results for all runs.
    device : torch.device | str
        Device to run computations on.
    result_key : str, optional
        Key under which to store the result (e.g., "flux_deflectometry", "flux_ideal"),
        by default "flux_deflectometry".
    """
    results_dict: Dict[str, Dict[str, np.ndarray]] = {}
    results_path = pathlib.Path(result_file)
    if results_path.exists():
        results_dict = torch.load(results_path, weights_only=False)

    scenario_h5_path = pathlib.Path(scenario_path).with_suffix(".h5")

    # Load scenario.
    with h5py.File(scenario_h5_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    scenario.light_sources.light_source_list[0].number_of_rays = 6000
    heliostat_properties_tuples: list[
        tuple[str, pathlib.Path] | tuple[str, pathlib.Path, pathlib.Path]
    ] = [
        (
            name,
            pathlib.Path(
                f"{paint_dir}/{name}/{config_dictionary.paint_properties_folder_name}/{name}{config_dictionary.paint_properties_file_name_ending}"
            ),
        )
        for name in heliostats
    ]

    # Extract facet translations and canting vectors (homogeneous 4D for downstream APIs).
    facet_transforms = extract_canting_and_translation_from_properties(
        heliostat_list=heliostat_properties_tuples,
        convert_to_4d=True,
        device=device,
    )
    facet_transforms_by_name = {
        heliostat_name: (facet_translations, facet_canting_vectors)
        for heliostat_name, facet_translations, facet_canting_vectors in facet_transforms
    }

    # Build properties list for calibration extraction.
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path]]] = []
    for name in heliostats:
        heliostat_data_mapping.append(
            (
                name,
                [
                    pathlib.Path(
                        f"{paint_dir}/{name}/Calibration/{MEASUREMENT_IDS[name]}-calibration-properties.json"
                    )
                ],
            )
        )

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
            for heliostat_name, calibration_properties_paths in heliostat_data_mapping
            if heliostat_name in scenario.heliostat_field.heliostat_groups[0].names
        ],
        heliostat_names=scenario.heliostat_field.heliostat_groups[0].names,
        target_area_names=scenario.target_areas.names,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

    # Perform alignment and ray tracing to generate flux density images.
    flux = align_and_trace_rays(
        scenario=scenario,
        aim_points=focal_spots_calibration,
        light_direction=incident_ray_directions_calibration,
        active_heliostats_mask=heliostats_mask_calibration,
        target_area_mask=target_area_mask_calibration,
        device=device,
    )

    # Load raw and UTIS image, update results.
    for i, name in enumerate(heliostats):
        if name not in results_dict:
            results_dict[name] = {}

        # Populate once if not present.
        if "image_raw" not in results_dict[name]:
            raw_image = load_image_as_tensor(
                name, str(paint_dir), MEASUREMENT_IDS[name], "cropped"
            )
            results_dict[name]["image_raw"] = raw_image.cpu().detach().numpy()

        if "utis_image" not in results_dict[name]:
            utis_image = load_image_as_tensor(
                name, str(paint_dir), MEASUREMENT_IDS[name], "flux"
            )
            results_dict[name]["utis_image"] = utis_image.cpu().detach().numpy()

        # Store flux for the given scenario under the provided key.
        results_dict[name][result_key] = flux[i].cpu().detach().numpy()

        # For the deflectometry scenario, keep the surface plot content as before.
        if result_key == "flux_deflectometry" and "surface" not in results_dict[name]:
            # Surface points per facet (canted).
            facet_points_canted = (
                scenario.heliostat_field.heliostat_groups[0]
                .surface_points[i]
                .view(4, 2500, 4)
            )

            # Apply inverse canting and translation using properties-derived transforms.
            facet_translations, facet_canting_vectors = facet_transforms_by_name[name]
            facet_points_decanted = perform_inverse_canting_and_translation(
                canted_points=facet_points_canted,
                translation=facet_translations,
                canting=facet_canting_vectors,
                device=device,
            )

            # Convert to Z grid for plotting.
            facet_points_decanted = (
                facet_points_decanted[:, :, 2].view(4, 50, 50).cpu().detach().numpy()
            )
            surface_z_grid = np.block(
                [
                    [facet_points_decanted[2], facet_points_decanted[0]],
                    [facet_points_decanted[3], facet_points_decanted[1]],
                ]
            )
            results_dict[name]["surface"] = surface_z_grid

    torch.save(results_dict, results_path)


def main():
    config = load_config()

    device = torch.device(config["device"])
    device = get_device(device)

    paint_dir = pathlib.Path(config["paint_repository_base_path"])
    paint_plots_base_path = pathlib.Path(config["base_path"])
    results_path = join_safe(
        paint_plots_base_path, config["results_raytracing_dict_path"]
    )
    # save_plot_path = join_safe(examples_base, config["examples_save_plot_path"])

    # device = get_device()
    # set_logger_config()
    # torch.manual_seed(7)
    # torch.cuda.manual_seed(7)

    scenario_base = join_safe(paint_plots_base_path, config["raytracing_scenario_path"])
    scenario_deflec_base = pathlib.Path(str(scenario_base) + "_deflectometry")
    scenario_ideal_base = pathlib.Path(str(scenario_base) + "_ideal")

    heliostats = ["AA39", "AY26", "BC34"]

    # Generate and merge flux images for both scenarios into one results file.
    generate_flux_images(
        scenario_path=scenario_deflec_base,
        heliostats=heliostats,
        paint_dir=str(paint_dir),
        result_file=str(results_path),
        device=device,
        result_key="flux_deflectometry",
    )
    generate_flux_images(
        scenario_path=scenario_ideal_base,
        heliostats=heliostats,
        paint_dir=str(paint_dir),
        result_file=str(results_path),
        device=device,
        result_key="flux_ideal",
    )


if __name__ == "__main__":
    main()
