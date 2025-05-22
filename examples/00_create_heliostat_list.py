import re
import pathlib
import json
from typing import Union, List


def find_heliostats_with_min_calibrations(
    paint_dir: Union[str, pathlib.Path],
    min_files: int = 100,
) -> List[str]:
    """
    Scans the PAINT directory for heliostat folders matching pattern [A-Z]{2}[0-9]{2]
    and returns those with at least `min_files` calibration JSONs and a deflectometry file.

    Parameters
    ----------
    paint_dir : str or Path
        Path to the base PAINT directory.
    min_files : int
        Minimum number of calibration files required.

    Returns
    -------
    List[str]
        Sorted list of valid heliostat names (e.g., ["AB26", "AY35", ...])
    """
    paint_dir = pathlib.Path(paint_dir)
    name_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}$")
    valid_heliostats = []

    for subdir in paint_dir.iterdir():
        if not subdir.is_dir():
            continue

        name = subdir.name
        if not name_pattern.match(name):
            continue

        calibration_dir = subdir / "Calibration"
        deflectometry_dir = subdir / "Deflectometry"
        if not calibration_dir.exists() or not deflectometry_dir.exists():
            continue

        calibration_files = list(calibration_dir.glob("*calibration-properties*.json"))
        if len(calibration_files) < min_files:
            continue

        deflectometry_files = sorted(deflectometry_dir.glob(f"{name}-filled*.h5"))
        if not deflectometry_files:
            continue

        valid_heliostats.append(name)

    return sorted(valid_heliostats)


def save_heliostat_list(heliostats: List[str], output_path: Union[str, pathlib.Path]) -> None:
    """
    Save list of heliostats to a JSON file.

    Parameters
    ----------
    heliostats : List[str]
        List of heliostat names.
    output_path : str or Path
        File path to save the list to.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(heliostats, f, indent=2)
    print(f"Saved {len(heliostats)} heliostat names to {output_path}")


if __name__ == "__main__":
    paint_dir = "/workVERLEIHNIX/share/PAINT"
    output_json = "examples/data/heliostat_list.json"
    min_calibrations = 100
    max_heliostats = 50
    exclude = {"AA39"}  # Known outlier(s)

    heliostats = find_heliostats_with_min_calibrations(paint_dir, min_files=min_calibrations)

    # Filter known outliers
    heliostats = [h for h in heliostats if h not in exclude]

    # Limit to top N
    heliostats = heliostats[:max_heliostats]

    print(f"Selected {len(heliostats)} heliostats:")
    print(heliostats)

    save_heliostat_list(heliostats, output_json)
