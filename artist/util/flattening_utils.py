from dataclasses import dataclass
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Generator, Tuple

#from artist.util.surface_converter import AnalyticalConfig

class SurfaceFlattener:
    """
    Flattens or unflattens batches of 3D surfaces.
    Modes:
      - 'flat': planar projection via ENU→XY basis rotation + translation.
      - 'curved': not yet implemented.
    """
    def __init__(
        self,
        surfaces
    ):

        from dataclasses import is_dataclass

        if not is_dataclass(surfaces):
            raise TypeError("Expected a dataclass instance for surfaces")

        if surfaces.flatten_mode == "flat":
            self.flatten_mode = "flat"

            canting_e = surfaces.canting_e[:,:3]
            canting_n = surfaces.canting_n[:,:3]
            ###TODO CANTING ADAPTION REMOVE LATER####
            canting_n = torch.cat((-canting_n[:, :2], -canting_n[:, 2:]), dim=1)
            canting_e = torch.cat((-canting_e[:, :2], -canting_e[:, 2:]), dim=1)

            self._canting_n= canting_n
            self._canting_e = canting_e
            print(self._canting_e)
            print(self._canting_n)

            self._translation = surfaces.facet_translation_vectors[:,:3]

            self._rotation         : torch.Tensor | None = None
            self._inverse_rotation : torch.Tensor | None = None
            self._offset           : torch.Tensor | None = None

            # Precompute all per-facet rotation matrices and offsets
            self._compute_flat_matrices()

        elif surfaces.flatten_mode == "curved":
            self.flatten_mode = "curved"
            self._curved_metadata = surfaces.metadata
            raise NotImplementedError("Curved flattening is not yet implemented")
        else:
            raise ValueError("Unknown surfaces dataclass")

    # ──────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────

    def _is_multi_facet(self, points: torch.Tensor) -> bool:
        """True if first dim matches the number of facets in this flattener."""
        return (points.ndim >= 2
                and points.shape[-1] == 3
                and points.shape[0] == self._rotation.shape[0])

    @contextmanager
    def _as_facet_first(
        self, points: torch.Tensor
    ) -> Generator[Tuple[torch.Tensor, Tuple[int, ...], bool], None, None]:
        """
        Yield `points` reshaped to (F, P, 3).
        Keeps track of the original shape so we can put it back afterwards.
        """
        if points.shape[-1] != 3:
            raise ValueError("last dimension must be 3")

        orig_shape   = points.shape                    # (… ,3)
        have_f_dim   = (points.ndim >= 3               # possible facet dim exists
                        and points.shape[0] == self._rotation.shape[0])

        if not have_f_dim:
            # treat as single facet → add dummy F-dimension
            points = points.unsqueeze(0)               # (1, … ,3)

        # now collapse everything except facet + coord
        F = points.shape[0]
        P = points[..., 0].numel() // F                # total pts per facet
        points = points.reshape(F, P, 3)               # (F,P,3)

        try:
            yield points, orig_shape, have_f_dim
        finally:
            pass   # nothing to clean up

    def _restore_shape(
        self,
        points_flat: torch.Tensor,
        orig_shape: Tuple[int, ...],
        had_f_dim: bool,
    ) -> torch.Tensor:
        """
        Inverse of `_as_facet_first`.
        """
        F, P, _ = points_flat.shape
        if not had_f_dim:
            # squeeze the dummy facet dim away
            points_flat = points_flat.squeeze(0)       # (P,3) or (M,M,3)
            new_shape   = orig_shape
        else:
            # replace point dimension(s) inside original shape
            spatial = orig_shape[1:-1]                 # everything but F & xyz
            points_flat = points_flat.view(orig_shape) # (F, … ,3)
            new_shape   = orig_shape
        return points_flat.reshape(new_shape)
    
    @staticmethod
    def _build_matrices(
        east: torch.Tensor, north: torch.Tensor, translation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (R, Rᵀ, offset) for a *single* facet.

        R columns  = [east, north, up]
        offset     = T · Rᵀ   (i.e. ENU origin expressed in XY coords)
        """
        east  = F.normalize(east,  dim=-1)                      # (3,)
        north = F.normalize(north, dim=-1)                      # (3,)
        up    = F.normalize(torch.cross(east, north), dim=-1)   # (3,)
        R     = torch.stack([east, north, up], dim=-1)          # (3,3)
        Rinv  = R.T                                             # (3,3)
        offset = (translation @ Rinv)                           # (3,)
        return R, Rinv, offset
    

    def _compute_flat_matrices(self) -> None:
        """
        Precompute per‐facet rotation matrices (ENU→XY and XY→ENU)
        and the fused translation offset.
        """
        # Normalize each ENU axis

        east_vectors  = F.normalize(self._canting_e, dim=1)            # (F,3)
        north_vectors = F.normalize(self._canting_n, dim=1)            # (F,3)
        up_vectors    = F.normalize(torch.cross(east_vectors,
                                                north_vectors,
                                                dim=1), dim=1)     # (F,3)

        # Build rotation matrices: columns are [east, north, up]
        rotation_matrices      = torch.stack([east_vectors,
                                              north_vectors,
                                              up_vectors], dim=2)   # (F,3,3)
        inverse_rotations      = rotation_matrices.transpose(1, 2)      # (F,3,3)

        # Compute fused offset: T @ (XY→ENU) for each facet
        offsets = torch.bmm(self._translation.unsqueeze(1),
                            inverse_rotations).squeeze(1)               # (F,3)

        # Store for use in flatten/unflatten
        self._rotation          = rotation_matrices
        self._inverse_rotation  = inverse_rotations
        self._offset            = offsets

    def _select_matrices(
        self,
        points      : torch.Tensor,
        *,
        flatten     : bool,
        facet_idx   : int | None = None,
        translation : torch.Tensor | None = None,
        canting_e   : torch.Tensor | None = None,
        canting_n   : torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Return (rot, trans, off) for either flatten (flatten=True) or
        unflatten (flatten=False).

        • rot  : ENU→XY (flatten)  or  XY→ENU (unflatten)
        • trans: facet translation vector
        • off  : fused offset  (needed only for flatten)
        """
        # branch-free choice of the pre-computed matrices --------------------------
        base_rot   = self._inverse_rotation if flatten else self._rotation
        base_off   = self._offset           if flatten else None  # unflatten ignores
        base_trans = self._translation

        if self._is_multi_facet(points):
            return base_rot, base_trans, base_off

        if facet_idx is not None:
            rot   = base_rot[facet_idx].unsqueeze(0)
            trans = base_trans[facet_idx].unsqueeze(0)
            off   = None if base_off is None else base_off[facet_idx].unsqueeze(0)
            return rot, trans, off

        if None not in (translation, canting_e, canting_n):
            R, Rinv, off = self._build_matrices(canting_e, canting_n, translation)
            rot = (Rinv if flatten else R).unsqueeze(0)
            trans = translation.unsqueeze(0)
            return rot, trans, off if flatten else None

        raise ValueError(
            "For a single-facet tensor, supply `facet_idx` or "
            "(`translation`, `canting_e`, `canting_n`)."
        )


    # ──────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────
    def flatten(self, points: torch.Tensor, **kw) -> torch.Tensor:
        if self.flatten_mode != "flat":
            raise NotImplementedError

        rot, trans, off = self._select_matrices(points, flatten=True, **kw)

        with self._as_facet_first(points) as (pts, orig_shape, had_f_dim):
            flattened = torch.matmul(pts, rot) - off.unsqueeze(1)
            return self._restore_shape(flattened, orig_shape, had_f_dim)


    def unflatten(self, points: torch.Tensor, **kw) -> torch.Tensor:
        if self.flatten_mode != "flat":
            raise NotImplementedError

        rot, trans, _ = self._select_matrices(points, flatten=False, **kw)

        with self._as_facet_first(points) as (pts, orig_shape, had_f_dim):
            unflattened = torch.matmul(pts, rot) + trans.unsqueeze(1)
        return self._restore_shape(unflattened, orig_shape, had_f_dim)

