import pathlib
import torch
from PIL import Image
import numpy as np

from artist.util.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util import config_dictionary,paint_loader

def find_latest_deflectometry_file(name, paint_dir):
    search_path = pathlib.Path(paint_dir) / name / "Deflectometry"
    pattern = f"{name}-filled*.h5"
    files = sorted(search_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No deflectometry file found for {name} in {search_path}")
    return files[-1]

def generate_paint_scenario(paint_dir, scenario_path, tower_file, heliostat_names, device="cpu"):

    # Validate scenario path
    scenario_path = pathlib.Path(scenario_path)
    if not scenario_path.parent.is_dir():
        raise FileNotFoundError(
            f"The folder ``{scenario_path.parent}`` to save the scenario does not exist."
        )

    # Load power plant and target area
    tower_file = pathlib.Path(tower_file)
    power_plant_config, target_area_list_config = paint_loader.extract_paint_tower_measurements(
        tower_measurements_path=tower_file,
        device=device,
    )

    # Define light source
    light_source = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=500,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )
    light_source_list_config = LightSourceListConfig(light_source_list=[light_source])

    # Extract receiver target area
    target_area = [
        ta for ta in target_area_list_config.target_area_list
        if ta.target_area_key == config_dictionary.target_area_receiver
    ]
    if not target_area:
        raise ValueError("No receiver target area found in tower measurements.")

    # Prepare heliostat files
    heliostat_files_list = [
        (
            name,
            pathlib.Path(f"{paint_dir}/{name}/Properties/{name}-heliostat-properties.json"),
            find_latest_deflectometry_file(name, paint_dir)
        )
        for name in heliostat_names
    ]

    # Load heliostats
    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        heliostat_and_deflectometry_paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        aim_point=target_area[0].center,
        device=device,
    )

    # Generate scenario
    scenario_generator = ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
    
    
def load_image_as_tensor(name: str, PAINT_DIR: str, measurements_id: int, key: str) -> torch.Tensor:
    """Load a flux PNG image as grayscale and return it as a torch tensor."""
    # Build the path
    image_path = pathlib.Path(f"{PAINT_DIR}/{name}/Calibration/{measurements_id}-{key}.png")
    
    # Open image in grayscale ('L' mode)
    image = Image.open(image_path).convert('L')
    
    # Convert to tensor and normalize to [0,1] float
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
        
    return tensor


def calculate_flux_deviation(f1, f2, normalize_by="f1"):
    """
    Calculates the flux deviation between two tensors.

    Args:
        f1 (torch.Tensor): First flux tensor (reference).
        f2 (torch.Tensor): Second flux tensor (comparison).
        normalize_by (str, optional): Specifies the normalization method.
                                       "f1" (default) - Normalizes by mean(f1)
                                       "f2" - Normalizes by mean(f2)

    Returns:
        torch.Tensor: Flux deviation.
    """
    
    
    # normalize
    f1 = torch.tensor(0.1*f1/f1.mean())
    f2 = torch.tensor( 0.1*f2/f2.mean())
    
    if f1.shape != f2.shape:
        raise ValueError(f"❌ ERROR: Shape mismatch: f1 {f1.shape} vs f2 {f2.shape}")

    abs_diff = torch.abs(f1 - f2)  # Element-wise absolute difference
    sum_abs_diff = torch.sum(abs_diff, dim=(-1, -2))  # Sum over last two dimensions (H, W)

    # Choose normalization factor
    norm_factor = torch.mean(f1 if normalize_by == "f1" else f2, dim=(-1, -2))
    norm_factor = torch.clamp(norm_factor, min=1e-8)  # Avoid division by zero

    # Compute flux deviation
    dev = sum_abs_diff / (norm_factor * f1.size(-1) * f1.size(-2))

    return dev