import enum
from typing import List, Optional, Tuple

import torch
import torch as th
from yacs.config import CfgNode

import utils


@enum.unique
class CantingAlgorithm(enum.Enum):
    STANDARD = 'standard'
    ACTIVE = 'active'


def get_algorithm(canting_cfg: CfgNode) -> CantingAlgorithm:
    canting_algo = next(
        (canting_cfg.ALGORITHM == algo.value for algo in CantingAlgorithm),
        None,
    )
    if canting_algo is None:
        raise ValueError('unknown canting algorithm')
    return canting_algo


def canting_enabled(canting_cfg: CfgNode) -> bool:
    return canting_cfg.FOCUS_POINT != 0


def get_focus_point(
        canting_cfg: CfgNode,
        receiver_center: Optional[torch.Tensor],
        # The normal of the ideal heliostat.
        ideal_normal: List[float],
        dtype: th.dtype,
        device: th.device,
) -> Optional[torch.Tensor]:
    if canting_cfg.FOCUS_POINT is not None:
        if isinstance(canting_cfg.FOCUS_POINT, list):
            assert len(canting_cfg.FOCUS_POINT) == 3, \
                'focus point as a list must be of length 3'
            focus_point: Optional[torch.Tensor] = th.tensor(
                canting_cfg.FOCUS_POINT,
                dtype=dtype,
                device=device,
            )
        # We explicitly don't check for the float type so that
        # distance can be given integers as well.
        elif canting_cfg.FOCUS_POINT != float('inf'):
            focus_point = th.tensor(
                ideal_normal,
                dtype=dtype,
                device=device,
            ) * canting_cfg.FOCUS_POINT
        else:
            focus_point = None
    else:
        assert receiver_center is not None
        focus_point = receiver_center
    return focus_point


def calc_focus_normal(
        receiver_center: torch.Tensor,
        heliostat_position_on_field: torch.Tensor,
        facet_position: torch.Tensor,
        facet_normal: torch.Tensor,
) -> torch.Tensor:
    hel_to_recv = receiver_center - heliostat_position_on_field
    focus_distance = th.linalg.norm(hel_to_recv)
    focus_point = facet_normal * focus_distance
    target_normal = focus_point - facet_position
    target_normal /= th.linalg.norm(target_normal)
    return target_normal


def get_focus_normal(
        focus_point: Optional[torch.Tensor],
        heliostat_position_on_field: torch.Tensor,
        facet_position: torch.Tensor,
        facet_normal: torch.Tensor,
        # The normal of the ideal heliostat.
        ideal_normal: List[float],
) -> torch.Tensor:
    if focus_point is not None:
        target_normal = calc_focus_normal(
            focus_point,
            heliostat_position_on_field,
            facet_position,
            facet_normal,
        )
    else:
        target_normal = th.tensor(
            ideal_normal,
            dtype=facet_position.dtype,
            device=facet_position.device,
        )
    return target_normal


def apply_rotation(
        rot_mat: torch.Tensor,
        values: torch.Tensor,
) -> torch.Tensor:
    return th.matmul(
        rot_mat.unsqueeze(0),
        values.unsqueeze(-1),
    ).squeeze(-1)


def cant_facet_to_normal(
        facet_position: torch.Tensor,
        start_normal: torch.Tensor,
        target_normal: torch.Tensor,
        discrete_points: torch.Tensor,
        discrete_points_ideal: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cant_rot = utils.get_rot_matrix(start_normal, target_normal)

    (
        discrete_points,
        discrete_points_ideal,
        normals,
        normals_ideal,
    ) = map(
        lambda t: apply_rotation(cant_rot, t),
        [
            discrete_points - facet_position,
            discrete_points_ideal - facet_position,
            normals,
            normals_ideal,
        ],
    )

    return (
        discrete_points + facet_position,
        discrete_points_ideal + facet_position,
        normals,
        normals_ideal,
    )


def cant_facet_to_point(
        heliostat_position_on_field: torch.Tensor,
        facet_position: torch.Tensor,
        focus_point: Optional[torch.Tensor],
        discrete_points: torch.Tensor,
        discrete_points_ideal: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
        # The normal of the ideal heliostat.
        ideal_normal: List[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    facet_normal = normals_ideal.mean(dim=0)
    facet_normal /= th.linalg.norm(facet_normal)

    target_normal = get_focus_normal(
        focus_point,
        heliostat_position_on_field,
        facet_position,
        facet_normal,
        ideal_normal,
    )

    return cant_facet_to_normal(
        facet_position,
        facet_normal,
        target_normal,
        discrete_points,
        discrete_points_ideal,
        normals,
        normals_ideal,
    )
