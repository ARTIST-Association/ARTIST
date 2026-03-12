import logging
import pathlib
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import torch
from PIL import Image
import torchvision

from artist.util import track_runtime, runtime_log
import artist.util.index_mapping
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the calibration data parser."""


class CalibrationDataParser:
    """
    Abstract base class for all calibration data parsers.

    Attributes
    ----------
    sample_limit : int | None
        The number specifying the maximum number of samples to be loaded.

    Methods
    -------
    parse_data_for_reconstruction()
        Extract measured fluxes and their respective calibration properties data.
    load_flux_from_png()
        Load flux density distributions as tensors from png images.
    """

    def __init__(
        self,
        sample_limit: int | None = None,
    ) -> None:
        """
        Initialize the the base data parser.

        Parameters
        ----------
        sample_limit : int | None
            The number specifying the maximum number of samples to be loaded (default is None).
        """
        self.sample_limit = sample_limit

    def parse_data_for_reconstruction(
        self,
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        heliostat_group: HeliostatGroup,
        scenario: Scenario,
        bitmap_resolution: torch.Tensor = torch.tensor(
            [
                artist.util.index_mapping.bitmap_resolution,
                artist.util.index_mapping.bitmap_resolution,
            ]
        ),
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Extract measured fluxes and their respective calibration properties data.

        Parameters
        ----------
        heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
            The mapping from heliostat to calibration data files.
        heliostat_group : HeliostatGroup
            The heliostat group.
        scenario : Scenario
            The scenario.
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction (default is torch.tensor([256,256])).
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        The measured flux density distributions.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        torch.Tensor
            The calibration focal spots.
            Tensor of shape [number_of_calibration_data_points, 4].
        torch.Tensor
            The incident ray directions.
            Tensor of shape [number_of_calibration_data_points, 4].
        torch.Tensor
            The motor positions.
            Tensor of shape [number_of_calibration_data_points, 2].
        torch.Tensor
            A mask with active heliostats and their replications.
            Tensor of shape [number_of_heliostats].
        torch.Tensor
            The target area mapping for the heliostats.
            Tensor of shape [number_of_active_heliostats].
        """
        raise NotImplementedError("Must be overridden!")

    def load_flux_from_png(
        self,
        heliostat_flux_path_mapping: list[tuple[str, list[pathlib.Path]]],
        heliostat_names: list[str],
        resolution: torch.Tensor = torch.tensor(
            [
                artist.util.index_mapping.bitmap_resolution,
                artist.util.index_mapping.bitmap_resolution,
            ]
        ),
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Load flux density distributions as tensors from png images.

        Note that the order of width and height are reversed in ``PIL`` and ``torch``.
        ``PIL`` takes (width, height), while ``torch`` tensors are [height, width].

        Parameters
        ----------
        heliostat_flux_path_mapping : list[tuple[str, list[pathlib.Path]]]
            The mapping of heliostats and their measured flux density distributions.
        heliostat_names : list[str]
            All possible heliostat names.
        resolution : torch.Tensor
            The resolution of the loaded png files (default is torch.tensor([256,256])).
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The measured flux density distributions.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        """
        device = get_device(device=device)
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        log.info(
            f"Rank {rank}: Beginning extraction of flux distributions from .png files."
        )

        width, height = resolution.to(device).tolist()
        path_mapping_dict = dict(heliostat_flux_path_mapping)
        
        total_number_of_measurements = sum(
            min(len(path_mapping_dict.get(name, [])), self.sample_limit or len(path_mapping_dict.get(name, [])))
            for name in heliostat_names
        )

        if total_number_of_measurements == 0:
            f"Rank {rank}: No measured flux density distributions were provided for this group."
            return torch.empty((0, height, width), device=device, dtype=torch.float32)

        measured_fluxes = torch.empty(
            (total_number_of_measurements, height, width),
            device=device,
            dtype=torch.float32,
        )

        index = 0
        for heliostat_name in heliostat_names:
            paths = path_mapping_dict.get(heliostat_name, [])
            number_of_measurements = min(len(paths), self.sample_limit or len(paths))

            for path in paths[:number_of_measurements]:
                bitmap_tensor = torchvision.io.read_image(
                    str(path), mode=torchvision.io.ImageReadMode.GRAY
                ).squeeze(0).float().to(device)

                if bitmap_tensor.shape != (height, width):
                    bitmap_data = Image.open(path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
                    bitmap_tensor = torch.from_numpy(np.asarray(bitmap_data, dtype=np.float32)).to(device)

                bitmap_tensor /= artist.util.index_mapping.bitmap_normalizer

                measured_fluxes[index] = bitmap_tensor
                index += 1

        log.info(
            f"Rank {rank}: Loading measured flux density distributions complete."
        )
        return measured_fluxes
