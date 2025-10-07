import logging
import pathlib
from collections import defaultdict
from typing import DefaultDict

import torch
from PIL import Image

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
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
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
        resolution: torch.Tensor = torch.tensor([256, 256]),
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

        flux_data_per_heliostat: DefaultDict[str, torch.Tensor] = defaultdict(
            torch.Tensor
        )

        width, height = resolution.to(device).tolist()

        total_number_of_measurements = 0
        for heliostat_name, paths in heliostat_flux_path_mapping:
            number_of_measurements = min(len(paths), self.sample_limit or len(paths))
            bitmaps = torch.empty(
                (number_of_measurements, height, width), device=device
            )
            for bitmap_index, path in enumerate(paths[:number_of_measurements]):
                bitmap_data = (
                    Image.open(path)
                    .convert("L")
                    .resize((width, height), Image.BILINEAR)
                )
                # Normalize pixel values from [0, 255] to [0.0, 1.0] (grayscale pixel values are in the range [0, 255]).
                bitmap_tensor = (
                    torch.tensor(bitmap_data.getdata(), device=device).view(
                        height, width
                    )
                    / 255.0
                )
                bitmaps[bitmap_index] = bitmap_tensor
                total_number_of_measurements = total_number_of_measurements + 1
            flux_data_per_heliostat[heliostat_name] = bitmaps

        measured_fluxes = torch.empty(
            (total_number_of_measurements, height, width), device=device
        )

        if total_number_of_measurements > 0:
            index = 0
            for name in heliostat_names:
                flux_list = flux_data_per_heliostat.get(name, [])
                for flux_data in flux_list:
                    n = flux_data.shape[0] if flux_data.ndim == 3 else 1
                    measured_fluxes[index : index + n] = flux_data
                    index += n

            log.info(
                f"Rank {rank}: Loading measured flux density distributions complete."
            )

        else:
            log.info(
                f"Rank {rank}: No measured flux density distributions were provided for this group."
            )

        return measured_fluxes
