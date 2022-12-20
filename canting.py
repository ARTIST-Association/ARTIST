from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Type, TYPE_CHECKING, TypeVar

import pytorch3d.transforms as throt
import torch
import torch as th
from yacs.config import CfgNode

if TYPE_CHECKING:
    from heliostat_models import AbstractHeliostat
import utils

S = TypeVar('S')


class CantingAlgorithm:
    name: str


class ActiveCantingAlgorithm(CantingAlgorithm):
    pass


class StandardCanting(CantingAlgorithm):
    name = 'standard'


class ActiveCanting(ActiveCantingAlgorithm):
    name = 'active'


class FirstSunCanting(ActiveCantingAlgorithm):
    name = 'first_sun'


class CantingParams:
    pass


@dataclass
class StandardCantingParams(CantingParams):
    focus_point: Optional[torch.Tensor]
    position_on_field: torch.Tensor


@dataclass
class FirstSunCantingParams(CantingParams):
    sun_direction: torch.Tensor
    focus_point: torch.Tensor
    position_on_field: torch.Tensor
    disturbance_angles: List[torch.Tensor]
    rotation_offset: torch.Tensor


def _subclass_tree(supertype: Type[S]) -> Set[Type[S]]:
    children = supertype.__subclasses__()
    # Avoid for-loop because we're growing the list in the loop.
    i = 0
    while i < len(children):
        child = children[i]
        children.extend(child.__subclasses__())
        i += 1
    return set(children)


def get_algorithm(canting_cfg: CfgNode) -> Optional[CantingAlgorithm]:
    canting_algo = next(
        (
            algo()
            for algo in _subclass_tree(CantingAlgorithm)
            if hasattr(algo, 'name') and canting_cfg.ALGORITHM == algo.name
        ),
        None,
    )
    if canting_algo is None:
        raise ValueError('unknown canting algorithm')

    if not canting_enabled(canting_cfg):
        return None
    return canting_algo


def canting_enabled(canting_cfg: CfgNode) -> bool:
    return canting_cfg.FOCUS_POINT != 0


def is_like_active(algo: Optional[CantingAlgorithm]) -> bool:
    return isinstance(algo, ActiveCantingAlgorithm)


def get_canting_params(
        heliostat: 'AbstractHeliostat',
        sun_direction: Optional[torch.Tensor],
) -> Optional[CantingParams]:
    if isinstance(heliostat.canting_algo, FirstSunCanting):
        assert sun_direction is not None, \
            'need sun direction to cant towards'
        assert heliostat.focus_point is not None, (
            'need focus point for perfectly canting towards the '
            'first sun'
        )
        canting_params: Optional[CantingParams] = FirstSunCantingParams(
            sun_direction,
            heliostat.focus_point,
            heliostat.position_on_field,
            heliostat.disturbance_angles,
            heliostat.rotation_offset,
        )
    elif (
            heliostat.canting_enabled
            and not is_like_active(heliostat.canting_algo)
    ):
        canting_params = StandardCantingParams(
            heliostat.focus_point,
            heliostat.position_on_field,
        )
    else:
        canting_params = None
    return canting_params


def get_focus_point(
        canting_cfg: CfgNode,
        heliostat_position_on_field: torch.Tensor,
        aim_point: Optional[torch.Tensor],
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
        # distance can be given as integers as well.
        elif canting_cfg.FOCUS_POINT != float('inf'):
            focus_point = heliostat_position_on_field + th.tensor(
                ideal_normal,
                dtype=dtype,
                device=device,
            ) * canting_cfg.FOCUS_POINT
        else:
            focus_point = None
    else:
        assert aim_point is not None
        focus_point = aim_point
    return focus_point


def calc_focus_normal(
        aim_point: torch.Tensor,
        heliostat_position_on_field: torch.Tensor,
        facet_position: torch.Tensor,
        facet_normal: torch.Tensor,
) -> torch.Tensor:
    hel_to_recv = aim_point - heliostat_position_on_field
    focus_distance = th.linalg.norm(hel_to_recv)
    facet_to_recv_distance = th.sqrt(
        focus_distance**2
        + th.linalg.norm(facet_position)**2
    )
    focus_point = facet_normal * facet_to_recv_distance
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


def _cant_facet_to_normal(
        facet_position: torch.Tensor,
        cant_rot: torch.Tensor,
        discrete_points: torch.Tensor,
        normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    discrete_points = apply_rotation(
        cant_rot, discrete_points - facet_position)
    normals = apply_rotation(cant_rot, normals)
    return (discrete_points + facet_position, normals)


def cant_facet_to_normal(
        facet_position: torch.Tensor,
        start_normal: torch.Tensor,
        target_normal: torch.Tensor,
        discrete_points: torch.Tensor,
        normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cant_rot = utils.get_rot_matrix(start_normal, target_normal)
    return _cant_facet_to_normal(
        facet_position,
        cant_rot,
        discrete_points,
        normals,
    )


def cant_facet_to_normal_with_ideal(
        facet_position: torch.Tensor,
        start_normal: torch.Tensor,
        target_normal: torch.Tensor,
        discrete_points: torch.Tensor,
        discrete_points_ideal: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cant_rot = utils.get_rot_matrix(start_normal, target_normal)

    discrete_points, normals = _cant_facet_to_normal(
        facet_position,
        cant_rot,
        discrete_points,
        normals,
    )
    discrete_points_ideal, normals_ideal = _cant_facet_to_normal(
        facet_position,
        cant_rot,
        discrete_points_ideal,
        normals_ideal,
    )
    return (
        discrete_points,
        discrete_points_ideal,
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

    return cant_facet_to_normal_with_ideal(
        facet_position,
        facet_normal,
        target_normal,
        discrete_points,
        discrete_points_ideal,
        normals,
        normals_ideal,
    )


def decant_facet(
        facet_position: torch.Tensor,
        facet_discrete_points: torch.Tensor,
        facet_discrete_points_ideal: torch.Tensor,
        facet_normals: torch.Tensor,
        facet_normals_ideal: torch.Tensor,
        # The normal of the ideal heliostat.
        ideal_normal: List[float],
        canting_params: Optional[CantingParams],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    orig_normal = facet_normals_ideal.mean(dim=0)
    orig_normal /= th.linalg.norm(orig_normal)

    target_normal = th.tensor(
        ideal_normal,
        dtype=facet_position.dtype,
        device=facet_position.device,
    )

    # De-cant so the facet is flat on z = 0.
    (
        facet_discrete_points,
        facet_discrete_points_ideal,
        facet_normals,
        facet_normals_ideal,
    ) = cant_facet_to_normal_with_ideal(
        facet_position,
        orig_normal,
        target_normal,
        facet_discrete_points,
        facet_discrete_points_ideal,
        facet_normals,
        facet_normals_ideal,
    )

    decanted_normal = facet_normals_ideal.mean(dim=0)
    decanted_normal /= th.linalg.norm(decanted_normal)

    if canting_params is None:
        canted_normal = orig_normal
    elif isinstance(canting_params, StandardCantingParams):
        canted_normal = get_focus_normal(
            canting_params.focus_point,
            canting_params.position_on_field,
            facet_position,
            decanted_normal,
            ideal_normal,
        )
    elif isinstance(canting_params, FirstSunCantingParams):
        from heliostat_models import heliostat_coord_system
        alignment = th.stack(heliostat_coord_system(
            facet_position + canting_params.position_on_field,
            canting_params.sun_direction,
            canting_params.focus_point,
            target_normal,
            canting_params.disturbance_angles,
            canting_params.rotation_offset,
        ))
        align_origin = throt.Rotate(alignment, dtype=alignment.dtype)
        facet_normals_ideal_rot = align_origin.transform_normals(
            facet_normals_ideal)
        canted_normal = facet_normals_ideal_rot.mean(dim=0)
    else:
        raise ValueError('encountered unhandled canting parameters')

    cant_rot = utils.get_rot_matrix(decanted_normal, canted_normal)
    return (
        facet_discrete_points,
        facet_discrete_points_ideal,
        facet_normals,
        facet_normals_ideal,
        cant_rot,
    )
