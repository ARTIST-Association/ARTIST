import torch
import torch.nn.functional as F
from artist.util import utils

class SurfaceFlattener:
    """
    Flattens (rolls out) a 3D surface onto the XY plane.

    Modes:
        - 'flat': simple planar projection using ENU basis rotation and translation.
        - 'curved': not yet implemented (planned: planar projection + custom flattening).
    """
    def __init__(
        self,
        flattening_mode: str = "flat"
    ):
        if flattening_mode not in ("flat", "curved"):
            raise ValueError("flattening_mode must be 'flat' or 'curved'")
        self.flattening_mode = flattening_mode

        if self.flattening_mode == "curved":
            raise NotImplementedError(
                "Curved flattening is not yet implemented"
            )

        # Cached per-facet matrices and offsets:
        self.flat_heliostat_rotations: torch.Tensor | None = None             # (F,3,3)
        self.flat_heliostat_inverse_rotations: torch.Tensor | None = None     # (F,3,3)
        self.flat_heliostat_inverse_offsets: torch.Tensor | None = None       # (F,3)

    def flatten(self, surface) -> None:
        """
        Flatten the given surface in-place by applying a fused translate+rotate per facet.
        """
        if self.flattening_mode != "flat":
            raise NotImplementedError("Curved flattening is not yet implemented")

        if self.flat_heliostat_inverse_offsets == None:
            self._prepare_flat_heliostat_matrices(surface)
        self._apply_flatten_heliostat(surface)

    def unflatten(self, surface) -> None:
        """
        Reverse the flattening (roll back to 3D) in-place by rotating then translating back.
        """
        if self.flattening_mode != "flat":
            raise NotImplementedError("Curved unflattening is not yet implemented")

        if self.flat_heliostat_inverse_offsets == None:
            self._prepare_flat_heliostat_matrices(surface)
        self._apply_unflatten_heliostat(surface)

    def _prepare_flat_heliostat_matrices(self, surface) -> None:
        """
        Precompute ENU->XY and XY->ENU rotations and the fused translation offsets.
        """
        rotation_matrices = []
        inverse_rotation_matrices = []
        inverse_offsets = []

        for facet in surface.facets:
            easting = F.normalize(facet.canting_e[:3], dim=0)
            northing = F.normalize(facet.canting_n[:3], dim=0)
            upping = F.normalize(torch.cross(easting, northing), dim=0)

            rotation_matrix = torch.stack([easting, northing, upping], dim=1)      # ENU->XY
            inverse_matrix = rotation_matrix.T                                     # XY->ENU

            # fused offset: translation_vector @ inverse_matrix
            fused_offset = facet.translation_vector[:3] @ inverse_matrix

            rotation_matrices.append(rotation_matrix)
            inverse_rotation_matrices.append(inverse_matrix)
            inverse_offsets.append(fused_offset)

        self.flat_heliostat_rotations = torch.stack(rotation_matrices, dim=0)
        self.flat_heliostat_inverse_rotations = torch.stack(inverse_rotation_matrices, dim=0)
        self.flat_heliostat_inverse_offsets = torch.stack(inverse_offsets, dim=0)

    def _apply_flatten_heliostat(self, surface) -> None:
        """
        For each facet: apply inverse rotation then subtract fused offset.
        """
        assert self.flat_heliostat_inverse_rotations is not None
        assert self.flat_heliostat_inverse_offsets is not None

        for index, facet in enumerate(surface.facets):
            inverse_rotation = self.flat_heliostat_inverse_rotations[index]
            offset = self.flat_heliostat_inverse_offsets[index]

            # rotate points (batch) and subtract offset in one fused operation per point
            rotated = facet.control_points[:, :3] @ inverse_rotation
            facet.control_points[:, :3] = rotated - offset

    def _apply_unflatten_heliostat(self, surface) -> None:
        """
        For each facet: rotate by ENU basis then reapply original translation.
        """
        assert self.flat_heliostat_rotations is not None

        for index, facet in enumerate(surface.facets):
            rotation_matrix = self.flat_heliostat_rotations[index]

            # rotate back
            restored = facet.control_points[:, :3] @ rotation_matrix
            # reapply translation
            facet.control_points[:, :3] = restored + facet.translation_vector[:3]
