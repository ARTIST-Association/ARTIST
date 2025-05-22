from typing import Union, Optional
from copy import deepcopy
from itertools import repeat
import torch
import logging
from artist.field.facets_nurbs import NurbsFacet
from artist.util.configuration_classes import SurfaceConfig, FacetConfig

log = logging.getLogger(__name__)

class Surface(torch.nn.Module):
    """
    Implement the surface module which contains a list of facets.

    Attributes
    ----------
    facets : List[Facet]
        A list of facets that comprise the surface of the heliostat.

    Methods
    -------
    get_surface_points_and_normals()
        Calculate all surface points and normals from all facets.
    forward()
        Specify the forward pass.
    """
    @staticmethod
    def _clone_with_new_up_direction(surface_config: SurfaceConfig, control_points_up_location: float) -> SurfaceConfig:
        new_surface_config = deepcopy(surface_config)
        for facet_config in new_surface_config.facet_list:
            facet_config.control_points[..., 2] = control_points_up_location
        return new_surface_config

    # ──────────────────────────────────────────────────────────────────────
    # Constructor
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        surface_config: SurfaceConfig | None = None,              # full profile
        surface_config_measured: SurfaceConfig | None = None,     # measured profile
        surface_config_ideal: SurfaceConfig | None = None         # ideal profile
    ) -> None:

        super().__init__()

        self.surface_config= surface_config
        self.surface_config_measured= surface_config_measured
        self.surface_config_ideal= surface_config_ideal
        # ────────────────────────────────────────────────────────────────

        full_profile_available      = isinstance(surface_config,          SurfaceConfig)
        measured_profile_available  = isinstance(surface_config_measured, SurfaceConfig)
        ideal_profile_available     = isinstance(surface_config_ideal,    SurfaceConfig)

        if not (full_profile_available or measured_profile_available or ideal_profile_available):
            raise ValueError(
                "At least one of surface_config, surface_config_measured or "
                "surface_config_ideal must be supplied."
            )

        # ────────────────────────────────────────────────────────────────
        # 1. Re-create any missing SurfaceConfig objects
        # ────────────────────────────────────────────────────────────────
        if not full_profile_available:
            surface_config = deepcopy(
                surface_config_ideal if ideal_profile_available else surface_config_measured
            )
            for facet_index, facet_configuration_full in enumerate(surface_config.facet_list):
                ideal_z_values = (
                    surface_config_ideal.facet_list[facet_index].control_points[..., 2]
                    if ideal_profile_available else 0
                )
                measured_z_values = (
                    surface_config_measured.facet_list[facet_index].control_points[..., 2]
                    if measured_profile_available else 0
                )
                facet_configuration_full.control_points[..., 2] = (
                    ideal_z_values + measured_z_values
                )
            full_profile_available = True
            log.info("Full surface profile reconstructed from measured and/or ideal profiles.")

        if not measured_profile_available:
            surface_config_measured = deepcopy(surface_config)
            for facet_index, facet_configuration_measured in enumerate(surface_config_measured.facet_list):
                full_z_values = surface_config.facet_list[facet_index].control_points[..., 2]
                ideal_z_values = (
                    surface_config_ideal.facet_list[facet_index].control_points[..., 2]
                    if ideal_profile_available else 0
                )
                facet_configuration_measured.control_points[..., 2] = full_z_values - ideal_z_values
            measured_profile_available = True
            log.info("Measured surface profile derived from full and ideal profiles.")

        if not ideal_profile_available:
            surface_config_ideal = deepcopy(surface_config)
            for facet_index, facet_configuration_ideal in enumerate(surface_config_ideal.facet_list):
                full_z_values = surface_config.facet_list[facet_index].control_points[..., 2]
                measured_z_values = surface_config_measured.facet_list[facet_index].control_points[..., 2]
                facet_configuration_ideal.control_points[..., 2] = full_z_values - measured_z_values
            ideal_profile_available = True
            log.info("Ideal surface profile derived from full and measured profiles.")

        # ────────────────────────────────────────────────────────────────
        # 2. Consistency check:  full == measured + ideal   (vectorised)
        # ────────────────────────────────────────────────────────────────
        for facet_full, facet_measured, facet_ideal in zip(
            surface_config.facet_list,
            surface_config_measured.facet_list,
            surface_config_ideal.facet_list,
        ):
            difference = (
                facet_full.control_points[..., 2]
                - (facet_measured.control_points[..., 2] + facet_ideal.control_points[..., 2])
            )
            if not (abs(difference) < 1e-8).all():
                raise ValueError(
                    "Internal inconsistency detected: "
                    "z-coordinates of full profile are not the sum of measured and ideal."
                )

        # ────────────────────────────────────────────────────────────────
        # 3. Collect control-point arrays
        # ────────────────────────────────────────────────────────────────
        control_points_list = []
        control_points_measured_list = []
        control_points_ideal_list = []

        for facet_index in range(len(surface_config.facet_list)):
            control_points_list.append(
                surface_config.facet_list[facet_index].control_points
            )
            control_points_measured_list.append(
                surface_config_measured.facet_list[facet_index].control_points
            )
            control_points_ideal_list.append(
                surface_config_ideal.facet_list[facet_index].control_points
            )

        # ────────────────────────────────────────────────────────────────
        # 4. Build NurbsFacet objects
        # ────────────────────────────────────────────────────────────────

        # Prototype (= full profile)
        self.facets = [
            NurbsFacet(
                degree_e=facet_config.degree_e,
                degree_n=facet_config.degree_n,
                number_eval_points_e=facet_config.number_eval_points_e,
                number_eval_points_n=facet_config.number_eval_points_n,
                translation_vector=facet_config.translation_vector,
                control_points = control_points_list[facet_index],
            )
            for facet_index, facet_config in enumerate(surface_config.facet_list)
        ]

        # Measured profile facets
        self.facets_measured = [
            NurbsFacet(
                degree_e=facet_config.degree_e,
                degree_n=facet_config.degree_n,
                number_eval_points_e=facet_config.number_eval_points_e,
                number_eval_points_n=facet_config.number_eval_points_n,
                translation_vector=facet_config.translation_vector,
                control_points = control_points_measured_list[facet_index]
            )
            for facet_index, facet_config in enumerate(surface_config_measured.facet_list)
        ]

        # Ideal profile facets (useful if you ever need them separately)
        self.facets_ideal = [
            NurbsFacet(
                degree_e=facet_config.degree_e,
                degree_n=facet_config.degree_n,
                number_eval_points_e=facet_config.number_eval_points_e,
                number_eval_points_n=facet_config.number_eval_points_n,
                translation_vector=facet_config.translation_vector,
                control_points = control_points_ideal_list[facet_index]
            )
            for facet_index, facet_config in enumerate(surface_config_ideal.facet_list)
        ]
    def _check_surface_config_consitency(self, surface_config_measured, surface_config_ideal):
        for i, (facet_config_measured, facet_config_ideal) in enumerate(zip(surface_config_measured, surface_config_ideal)):
                attributes_to_compare = [
                    'degree_e', 'degree_n',
                    'number_eval_points_e', 'number_eval_points_n',
                    'translation_vector'
                ]
                for attribute in attributes_to_compare:
                    value_measured = getattr(facet_config_measured, attribute)
                    value_ideal = getattr(facet_config_ideal, attribute)
                    if isinstance(value_measured, torch.Tensor):
                        if not torch.equal(value_measured, value_ideal):
                            raise ValueError(f"Mismatch in {attribute} at index {i}. Can not load two facets with different heliostat properties.")
                    else:
                        if value_measured != value_ideal:
                            raise ValueError(f"Mismatch in {attribute} at index {i}. Can not load two facets with different heliostat properties.")
    def get_surface_points_and_normals(
        self, 
        facet_type: Optional[str] = None, 
        device: Union[torch.device, str] = "cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate all surface points and normals from all facets.

        Parameters
        ----------
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface normals.
        """
        device = torch.device(device)
        if facet_type is None or "fulL":
            facets = self.facets
        elif facet_type == "measured":
            facets = self.facets_measured
        elif facet_type == "ideal":
            facets = self.facets_ideal
        else:
            raise ValueError(f"Unknown facet type: {facet_type}. Must be 'full', 'measured' or 'ideal'. None value is treated as 'full'.")
        eval_point_per_facet = (
            facets[0].number_eval_points_n * facets[0].number_eval_points_e
        )
        surface_points = torch.empty(
            len(facets), eval_point_per_facet, 4, device=device
        )
        surface_normals = torch.empty(
            len(facets), eval_point_per_facet, 4, device=device
        )
        for i, facet in enumerate(facets):
            facet_surface = facet.create_nurbs_surface(device=device)
            (
                facet_points,
                facet_normals,
            ) = facet_surface.calculate_surface_points_and_normals(device=device)
            surface_points[i] = facet_points + facet.translation_vector
            surface_normals[i] = facet_normals
        return surface_points, surface_normals
    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
