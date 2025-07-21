import logging
import pathlib
from collections import defaultdict
from typing import DefaultDict

import torch
import torchvision.transforms as transforms
from PIL import Image

from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the flux distribution loader."""


def load_flux_from_png(
    heliostat_flux_path_mapping: list[tuple[str, list[pathlib.Path]]],
    heliostat_names: list[str],
    resolution: torch.Tensor = torch.tensor([256, 256]),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Load flux density distributions as tensors from png images.

    Parameters
    ----------
    heliostat_flux_path_mapping : list[tuple[str, list[pathlib.Path]]]
        The mapping of heliostats and their measured flux density distributions.
    heliostat_names : list[str]
        All possible heliostat names.
    resolution : torch.Tensor
        The resolution of the loaded png files (default is torch.tensor([256, 256])).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The measured flux density distributions.
    """
    device = get_device(device=device)

    log.info("Beginning extraction of flux distributions from .png files.")

    flux_data_per_heliostat: DefaultDict[str, torch.Tensor] = defaultdict(torch.Tensor)

    resolution = tuple(resolution.to(device).tolist())
    transform = transforms.Compose(
        [transforms.Resize(resolution), transforms.ToTensor()]
    )

    for heliostat_name, paths in heliostat_flux_path_mapping:
        bitmaps = torch.empty((len(paths), resolution[0], resolution[1]), device=device)
        for bitmap_index, path in enumerate(paths):
            bitmap_data = Image.open(path).convert("L")
            bitmap_tensor = transform(bitmap_data)
            bitmap = bitmap_tensor.squeeze(0)
            bitmaps[bitmap_index] = bitmap
        flux_data_per_heliostat[heliostat_name] = bitmaps

    total_number_of_measurements = sum(
        len(path_list) for _, path_list in heliostat_flux_path_mapping
    )

    measured_fluxes = torch.empty(
        (total_number_of_measurements, resolution[0], resolution[1]), device=device
    )

    if total_number_of_measurements > 0:
        index = 0
        for name in heliostat_names:
            for flux_data in flux_data_per_heliostat.get(name, []):
                measured_fluxes[index : index + flux_data.shape[0]] = flux_data
                index += 1

        log.info("Loading measured flux density distributions complete.")

    else:
        log.info("No measured flux density distributions were provided for this group.")

    return measured_fluxes
