import os
from typing import List, Optional, Tuple

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from environment import Environment
from heliostat_models import AbstractHeliostat
from render import Renderer
import utils
import pytorch3d.transforms as throt

def create_target(
        H: AbstractHeliostat,
        ENV: Environment,
        sun_direction: torch.Tensor,
        save_path: Optional[str] = None,
) -> torch.Tensor:
    device = H.device
    if save_path:
        target_save_data = (
            H.position_on_field,
            th.tensor(
                H.cfg.IDEAL.NORMAL_VECS,
                dtype=th.get_default_dtype(),
                device=device,
            ),
            H.discrete_points,
            H.normals,
            None,  # TODO

            ENV.receiver_center,
            ENV.receiver_plane_x,
            ENV.receiver_plane_y,
            ENV.receiver_plane_normal,
            None,  # TODO

            sun_direction,
            ENV.sun.num_rays,
            ENV.sun.mean,
            ENV.sun.cov,
        )

    H_aligned = H.align(sun_direction, ENV.receiver_center)
    R = Renderer(H_aligned, ENV)
    if save_path:
        if R.redraw_random_variables:
            xi = None
            yi = None
        else:
            xi = R.xi
            yi = R.yi

        utils.save_target(
            *(
                target_save_data
                + (
                    xi,
                    yi,

                    # We need the heliostat to be aligned here.
                    H_aligned.get_ray_directions(),
                    H_aligned.discrete_points,
                    save_path,
                )
            )
        )

    # Render Step
    # ===========
    target_bitmap = R.render()
    assert isinstance(target_bitmap, th.Tensor)
    return target_bitmap


@th.no_grad()
def generate_dataset(
        H: AbstractHeliostat,
        ENV: Environment,
        sun_directions: torch.Tensor,
        save_dir: Optional[str],
        writer: Optional[SummaryWriter] = None,
        prefix: str = '',
) -> torch.Tensor:

    if save_dir:
        save_path: Optional[str] = os.path.join(save_dir, 'target.pt')
    else:
        save_path = None

    device = H.device

    targets = None
    for (i, sun_direction) in enumerate(sun_directions):
        target_bitmap = create_target(
            H,
            ENV,
            sun_direction,
            save_path=save_path,
        )
        if targets is None:
            targets = th.empty(
                (len(sun_directions),) + target_bitmap.shape,
                dtype=th.get_default_dtype(),
                device=device,
            )
        targets[i] = target_bitmap
        if writer:
            writer.add_image(
                f"{prefix}target_{i}/originals",
                utils.colorize(target_bitmap),
            )
    assert targets is not None
    return targets


def _random_sun_array(
        cfg: CfgNode,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sun_directions = th.rand(
        (cfg.NUM_SAMPLES, 3),
        dtype=th.get_default_dtype(),
        device=device,
    )

    # Allow negative x- and y-values.
    sun_directions[:, :-1] -= 0.5
    sun_directions = (
        sun_directions
        / th.linalg.norm(sun_directions, dim=1).unsqueeze(-1)
    )
    ae = utils.vec_to_ae(sun_directions)
    return sun_directions, ae


def _grid_sun_array(
        cfg: CfgNode,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # create azi ele range space
    azi: List[float] = cfg.AZI_RANGE
    azi = th.linspace(
        azi[0],
        azi[1],
        int(azi[2]),
        dtype=th.get_default_dtype(),
        device=device,
    )

    ele: List[float] = cfg.ELE_RANGE
    ele = th.linspace(
        ele[0],
        ele[1],
        int(ele[2]),
        dtype=th.get_default_dtype(),
        device=device,
    )
    # all possible combinations of azi ele
    ae = th.cartesian_prod(azi, ele)
    # create 3D vector from azi, ele
    sun_directions = utils.ae_to_vec(ae[:, 0], ae[:, 1])
    return sun_directions, ae


def _vec_sun_array(
        cfg: CfgNode,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sun_directions: List[List[float]] = (
        utils.with_outer_list(cfg.DIRECTIONS))

    sun_directions: torch.Tensor = th.tensor(
        sun_directions,
        dtype=th.get_default_dtype(),
        device=device,
    )

    sun_directions = (
        sun_directions
        / th.linalg.norm(sun_directions, dim=1).unsqueeze(-1)
    )

    ae = utils.vec_to_ae(sun_directions, device=device)
    return sun_directions, ae

def _spheric_sun_array(
        cfg: CfgNode,
        device: th.device,
        train_vec: th.Tensor):
    
        if train_vec == None:
            raise(Exception("train_vec is None. Spheric testing needs a train vector"))
        if not train_vec.squeeze().shape ==th.Size([3]):
            raise(Exception("multiple sun vectors detected. spheric plot is only possible using 1 vector"))
        train_vec = train_vec  / th.linalg.norm(train_vec, dim=1).unsqueeze(1)
        ae = utils.vec_to_ae(train_vec, device).squeeze()
        ele_angles = th.linspace(0.1*180, 0.9*180, cfg.NUM_SAMPLES, dtype=th.get_default_dtype(), device=device)
        azi_angles = th.linspace(0, 180, cfg.NUM_SAMPLES, dtype=th.get_default_dtype(), device=device)
        
        north = th.tensor([[0.,1.,0.]], dtype=th.get_default_dtype(), device=device)
        t_ele = throt.Transform3d(device=device).rotate_axis_angle(ele_angles, "X").rotate_axis_angle(-ae[0], "Z")
        spheric_vecs_ele = t_ele.transform_normals(north).squeeze()
        
        other_direction = train_vec.clone()
        t_azi_western = throt.Transform3d(device=device).rotate_axis_angle(azi_angles, "Z")
        spheric_vecs_azi_west = t_azi_western.transform_normals(other_direction).squeeze()
        t_azi_eastern = throt.Transform3d(device=device).rotate_axis_angle(-azi_angles,  "Z")
        spheric_vecs_azi_east = t_azi_eastern.transform_normals(other_direction).squeeze()



        ae_azi_west_dir = utils.vec_to_ae(spheric_vecs_azi_west, device).detach().cpu()
        ae_azi_east_dir = utils.vec_to_ae(spheric_vecs_azi_east, device).detach().cpu()
        ae_ele_dir = th.nan_to_num(utils.vec_to_ae(spheric_vecs_ele, device)).detach().cpu()
        
        ae = th.cat([ae_azi_west_dir,ae_azi_east_dir,ae_ele_dir],0)
        sun_directions = th.cat([spheric_vecs_azi_west, spheric_vecs_azi_east, spheric_vecs_ele],0)

        return sun_directions, ae


def generate_sun_array(
        cfg_sun_directions: CfgNode,
        device: th.device,
        sun_direction: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = cfg_sun_directions
    case: str = cfg.CASE
    if case == "random":
        sun_directions, ae = _random_sun_array(cfg.RAND, device)
    elif case == "grid":
        sun_directions, ae = _grid_sun_array(cfg.GRID, device)
    elif case == "vecs":
        sun_directions, ae = _vec_sun_array(cfg.VECS, device)

    elif case == "spheric":
            sun_directions, ae = _spheric_sun_array(cfg.SPHERIC, device, sun_direction)
    else:
        raise ValueError("unknown `cfg.CASE` in `generate_sun_rays`")
    return sun_directions, ae
