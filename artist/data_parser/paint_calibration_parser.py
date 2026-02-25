import json
import logging
import pathlib
from collections import Counter, defaultdict

import paint.util.paint_mappings as paint_mappings
import torch

import artist.util.index_mapping
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the paint calibration data parser."""


class PaintCalibrationDataParser(CalibrationDataParser):
    """
    A calibration data parser for the data source ``PAINT``.

    Attributes
    ----------
    sample_limit : int | None
        The number specifying the maximum number of samples to be loaded.
    centroid_extraction_method : str
        The method by which the focal spot centroid is extracted.

    Methods
    -------
    parse_data_for_reconstruction()
        Extract measured fluxes and their respective calibration properties data.
    load_flux_from_png()
        Load flux density distributions as tensors from png images.

    See Also
    --------
    :class:`CalibrationDataParser` : Reference to the parent class.
    """

    def __init__(
        self,
        sample_limit: int | None = None,
        centroid_extraction_method: str = paint_mappings.UTIS_KEY,
    ) -> None:
        """
        Initialize the the paint calibration data parser.

        Parameters
        ----------
        sample_limit : int | None
            The number specifying the maximum number of samples to be loaded (default is None).
        centroid_extraction_method : str
            The method by which the focal spot centroid was extracted (default is the centroid extracted by ``UTIS``).
        """
        super().__init__(sample_limit=sample_limit)

        if centroid_extraction_method not in [
            paint_mappings.UTIS_KEY,
            paint_mappings.HELIOS_KEY,
        ]:
            raise ValueError(
                f"The selected centroid extraction method {centroid_extraction_method} is not yet supported. Please use either {paint_mappings.UTIS_KEY} or {paint_mappings.HELIOS_KEY}!"
            )
        self.centroid_extraction_method = centroid_extraction_method

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
        device = get_device(device=device)

        heliostat_flux_path_mapping = []
        heliostat_calibration_mapping = []

        for heliostat, path_properties, path_pngs in heliostat_data_mapping:
            if heliostat in heliostat_group.names:
                heliostat_flux_path_mapping.append((heliostat, path_pngs))
                heliostat_calibration_mapping.append((heliostat, path_properties))

        measured_flux_distributions = self.load_flux_from_png(
            heliostat_flux_path_mapping=heliostat_flux_path_mapping,
            heliostat_names=heliostat_group.names,
            resolution=bitmap_resolution,
            device=device,
        )

        (
            focal_spots,
            incident_ray_directions,
            motor_positions,
            active_heliostats_mask,
            target_area_mask,
        ) = self._parse_calibration_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )

        return (
            measured_flux_distributions,
            focal_spots,
            incident_ray_directions,
            motor_positions,
            active_heliostats_mask,
            target_area_mask,
        )

    def _parse_calibration_data(
        self,
        heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
        heliostat_names: list[str],
        target_area_names: list[str],
        power_plant_position: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract calibration data from ``PAINT`` calibration files.

        Parameters
        ----------
        heliostat_calibration_mapping : list[tuple[str, list[pathlib.Path]]]
            The mapping of heliostats and their calibration data files.
        power_plant_position : torch.Tensor
            The power plant position.
            Tensor of shape [3].
        heliostat_names : list[str]
            All possible heliostat names.
        target_area_names : list[str]
            All possible target area names.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
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
        device = get_device(device=device)

        log.info("Beginning extraction of calibration properties data from PAINT file.")

        target_indices = {name: index for index, name in enumerate(target_area_names)}

        # Gather calibration data.
        replication_counter: Counter[str] = Counter()
        calibration_data_per_heliostat = defaultdict(list)

        for heliostat_name, paths in heliostat_calibration_mapping:
            number_of_measurements = min(len(paths), self.sample_limit or len(paths))
            for path in paths[:number_of_measurements]:
                with open(path, "r") as f:
                    calibration_data_dict = json.load(f)
                replication_counter[heliostat_name] += 1

                calibration_data_per_heliostat[heliostat_name].append(
                    [
                        target_indices[
                            calibration_data_dict[paint_mappings.TARGET_NAME_KEY]
                        ],
                        calibration_data_dict[paint_mappings.FOCAL_SPOT_KEY][
                            self.centroid_extraction_method
                        ],
                        calibration_data_dict[paint_mappings.SUN_AZIMUTH],
                        calibration_data_dict[paint_mappings.SUN_ELEVATION],
                        [
                            calibration_data_dict[paint_mappings.MOTOR_POS_KEY][
                                paint_mappings.AXIS1_MOTOR_SAVE
                            ],
                            calibration_data_dict[paint_mappings.MOTOR_POS_KEY][
                                paint_mappings.AXIS2_MOTOR_SAVE
                            ],
                        ],
                    ]
                )

        total_samples = sum(replication_counter[name] for name in heliostat_names)
        calibration_replications = torch.tensor(
            [replication_counter[name] for name in heliostat_names],
            dtype=torch.int32,
            device=device,
        )

        target_area_mapping = torch.empty(
            total_samples, device=device, dtype=torch.long
        )
        focal_spots_global = torch.empty((total_samples, 3), device=device)
        azimuths = torch.empty(total_samples, device=device)
        elevations = torch.empty(total_samples, device=device)
        motor_positions = torch.empty((total_samples, 2), device=device)

        index = 0
        for name in heliostat_names:
            for (
                target_index,
                focal_spot,
                azimuth,
                elevation,
                motor_pos,
            ) in calibration_data_per_heliostat.get(name, []):
                target_area_mapping[index] = target_index
                focal_spots_global[index] = torch.tensor(focal_spot, device=device)
                azimuths[index] = azimuth
                elevations[index] = elevation
                motor_positions[index] = torch.tensor(motor_pos, device=device)
                index += 1

        focal_spots_enu = utils.convert_wgs84_coordinates_to_local_enu(
            focal_spots_global, power_plant_position, device=device
        )
        focal_spots = utils.convert_3d_points_to_4d_format(
            focal_spots_enu, device=device
        )

        light_source_positions_enu = utils.azimuth_elevation_to_enu(
            azimuths, elevations, degree=True, device=device
        )
        light_source_positions = utils.convert_3d_points_to_4d_format(
            light_source_positions_enu, device=device
        )
        incident_ray_directions = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - light_source_positions
        )

        log.info("Loading calibration properties data complete.")

        return (
            focal_spots,
            incident_ray_directions,
            motor_positions,
            calibration_replications,
            target_area_mapping,
        )
