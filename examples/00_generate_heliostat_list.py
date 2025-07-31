import re
import pathlib
import json
from typing import Union, List, Tuple

from artist.util import config_dictionary


def find_heliostats_with_min_calibrations(
    paint_dir: Union[str, pathlib.Path],
    min_files: int = 100,
    max_heliostats: int = 10,
    flux_suffix: str = "flux-centered",  # NEW PARAMETER
) -> List[Tuple[str, List[pathlib.Path], List[pathlib.Path]]]:
    """
    Scans the PAINT directory for heliostat folders and returns those with
    calibration JSONs and flux image files matching the given suffix.

    Parameters
    ----------
    paint_dir : str or Path
        Path to the base PAINT directory.
    min_files : int
        Minimum number of calibration files required.
    flux_suffix : str
        Suffix to use for flux image filenames (e.g., "flux", "flux-centered").

    Returns
    -------
    List[Tuple[str, List[Path], List[Path]]]
        Valid heliostat data.
    """
    paint_dir = pathlib.Path(paint_dir)
    name_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}$")
    heliostat_data = []

    for subdir in paint_dir.iterdir():
        if not subdir.is_dir():
            continue

        name = subdir.name
        if not name_pattern.match(name):
            continue

        calibration_dir = subdir / "Calibration"
        # deflectometry_dir = subdir / "Deflectometry"
        if not calibration_dir.exists(): #or not deflectometry_dir.exists():
             continue


        calibration_files = sorted(calibration_dir.glob("*calibration-properties*.json"))
        valid_calibration_files = []
        for path in calibration_files:
            try:
                with open(path, "r") as file:
                    calibration_dict = json.load(file)
                    focal_data = calibration_dict.get(config_dictionary.paint_focal_spot, {})
                    if (
                        config_dictionary.paint_helios in focal_data
                        and config_dictionary.paint_utis in focal_data
                    ):
                        valid_calibration_files.append(path)
            except Exception as e:
                print(f"Warning: Skipping {path} due to error: {e}")

        
        if len(valid_calibration_files) < min_files:
            continue

        valid_calibration_files = valid_calibration_files[:min_files]
        # deflectometry_files = sorted(deflectometry_dir.glob(f"{name}-filled*.h5"))
        # if not deflectometry_files:
        #     continue

        # Match flux images based on suffix (e.g., "flux", "flux-centered")
        flux_image_files = []
        for calib_path in valid_calibration_files:
            stem = calib_path.stem.replace("-calibration-properties", "")
            image_filename = f"{stem}-{flux_suffix}.png"
            image_path = calibration_dir / image_filename
            if image_path.exists():
                flux_image_files.append(image_path)

        heliostat_data.append((name, valid_calibration_files, flux_image_files))
        print(f"Added heliostat {name} to list. List length = {len(heliostat_data)}")
        if len(heliostat_data) >= max_heliostats:
            break

    return sorted(heliostat_data, key=lambda x: x[0])


def save_heliostat_list(heliostats: List[Tuple[str, List[pathlib.Path], List[pathlib.Path]]], output_path: Union[str, pathlib.Path]) -> None:
    """
    Save list of heliostats to a JSON file, converting Path objects to strings.

    Parameters
    ----------
    heliostats : List[Tuple[str, List[Path], List[Path]]]
        List of heliostat data with calibration and flux paths.
    output_path : str or Path
        File path to save the list to.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    serializable_data = [
        {
            "name": name,
            "calibrations": [str(p) for p in calibs],
            "flux_images": [str(p) for p in fluxes],
        }
        for name, calibs, fluxes in heliostats
    ]

    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=2)
    print(f"Saved {len(serializable_data)} heliostat entries to {output_path}")



if __name__ == "__main__":
    paint_dir = "/workVERLEIHNIX/share/PAINT"
    output_json = "examples/data/heliostat_files.json"
    min_calibrations = 80
    max_heliostats = 800
    exclude = {"AA39"}
    flux_suffix = "flux" 

    heliostat_data = find_heliostats_with_min_calibrations(
        paint_dir,
        min_files=min_calibrations,
        max_heliostats=max_heliostats,
        flux_suffix=flux_suffix,
    )

    heliostat_data = [h for h in heliostat_data if h[0] not in exclude]
    heliostat_data = heliostat_data[:max_heliostats]

    print(f"Selected {len(heliostat_data)} heliostats:")
    for name, calibs, fluxes in heliostat_data:
        print(f"- {name}: {len(calibs)} calibrations, {len(fluxes)} flux images ({flux_suffix})")

    save_heliostat_list(heliostat_data, output_json)
