import os
from typing import List, Optional, Tuple

import pytorch3d.transforms as throt
import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode
import random

from environment import Environment
from heliostat_models import AbstractHeliostat
from render import Renderer
import utils


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

    device = H.device
    save_path: Optional[str] = None

    targets = None
    for (i, sun_direction) in enumerate(sun_directions):
        if save_dir:
            save_path = os.path.join(
                save_dir,
                f'{prefix}_target_{i}.pt',
            )

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
    assert targets is not None
    log_dataset(writer, targets, prefix=prefix)
    return targets


def _random_sun_array(
        cfg: CfgNode,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = cfg.NUM_SAMPLES
    observerLatitude = cfg.LATITUDE
    observerLongitude = cfg.LONGITUDE

    ae = []
    for i in range(size):
        year = random.randint(1970, 2050)
        month = random.randint(1, 12)
        # Exclude late days in month, just cause a lot of if-clauses.
        day = random.randint(1, 29)
        hour = random.randint(1, 23)
        minute = random.randint(1, 59)
        sec = random.randint(1, 59)
        azi, ele = utils.calculateSunAngles(
            hour,
            minute,
            sec,
            day,
            month,
            year,
            observerLatitude,
            observerLongitude,
        )
        ae.append([azi, ele])

    ae = th.tensor(
        ae,
        dtype=th.get_default_dtype(),
        device=device,
    )
    sun_directions = utils.ae_to_vec(ae[:, 0], ae[:, 1])
    sun_directions = (
        sun_directions
        / th.linalg.norm(sun_directions, dim=1).unsqueeze(-1)
    )
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

    ae = utils.vec_to_ae(sun_directions)
    return sun_directions, ae


def _spheric_sun_array(
        cfg: CfgNode,
        device: th.device,
        train_vec: Optional[th.Tensor],
):
    if train_vec is None:
        raise ValueError(
            "train_vec is None. Spheric testing needs a train vector")
    if not train_vec.squeeze().shape == (3,):
        raise ValueError(
            "multiple sun vectors detected. "
            "spheric plot is only possible using 1 vector"
        )
    train_vec = train_vec / th.linalg.norm(train_vec, dim=1).unsqueeze(1)
    ae = utils.vec_to_ae(train_vec).squeeze()
    ele_angles = th.linspace(
        0.1 * 180,
        0.9 * 180,
        cfg.NUM_SAMPLES,
        dtype=th.get_default_dtype(),
        device=device,
    )
    azi_angles = th.linspace(
        0,
        180,
        cfg.NUM_SAMPLES,
        dtype=th.get_default_dtype(),
        device=device,
    )

    north = th.tensor([[0, 1, 0]], dtype=th.get_default_dtype(), device=device)
    t_ele = throt.Transform3d(device=device) \
                 .rotate_axis_angle(ele_angles, "X") \
                 .rotate_axis_angle(-ae[0], "Z")
    spheric_vecs_ele = t_ele.transform_normals(north).squeeze()

    other_direction = train_vec.clone()
    t_azi_western = throt.Transform3d(device=device) \
                         .rotate_axis_angle(azi_angles, "Z")
    spheric_vecs_azi_west = t_azi_western.transform_normals(
        other_direction).squeeze()
    t_azi_eastern = throt.Transform3d(device=device) \
                         .rotate_axis_angle(-azi_angles,  "Z")
    spheric_vecs_azi_east = t_azi_eastern.transform_normals(
        other_direction).squeeze()

    ae_azi_west_dir = utils.vec_to_ae(spheric_vecs_azi_west).detach().cpu()
    ae_azi_east_dir = utils.vec_to_ae(spheric_vecs_azi_east).detach().cpu()
    ae_ele_dir = th.nan_to_num(
        utils.vec_to_ae(spheric_vecs_ele)).detach().cpu()

    ae = th.cat([ae_azi_west_dir, ae_azi_east_dir, ae_ele_dir], 0)
    sun_directions = th.cat(
        [spheric_vecs_azi_west, spheric_vecs_azi_east, spheric_vecs_ele], 0)

    return sun_directions, ae


def _season_sun_array(
        device,
        stepsize_hours=1,
        stepsize_minutes=30,
        longest_day=True,
        shortest_day=True,
        equinox_spring=True,
        measurement_day=True,
        *measurement_date,
):
    """Input:
        - stepsize for defining the grid distances
        - device
        - which days to simulate
        - args must be in descending order years, months, days, ...

   Output:
       - List of all sun positions including simualted hours (N, 8)
         including [hour,minute,sec,day,month,year,azi,ele] starting
         with longest day, start of autumn, shortest day, start of
         spring, measurment day
       - length of different day simulations in shape (N,)
       - name of simulated days
    """
    minutes = list(range(0, 60, stepsize_minutes))
    trajectories = []
    infos = {}
    infos["date_time_ae"] = []
    if measurement_day:
        if len(measurement_date) >= 1:
            years = measurement_date[0]
        else:
            years = [2021]
        if len(measurement_date) >= 2:
            months = measurement_date.pop(1)
        else:
            months = [10]
        if len(measurement_date) >= 3:
            days = measurement_date.pop(2)
        else:
            days = [28]
        if len(measurement_date) >= 4:
            hours = measurement_date.pop(3)
        else:
            hours = list(range(8, 16, stepsize_hours))
        if len(measurement_date) >= 5:
            minutes = measurement_date.pop(4)
        else:
            minutes = list(range(0, 60, stepsize_minutes))
        if len(measurement_date) >= 6:
            secs = [measurement_date.pop(5)]
        else:
            secs = [0]
        if len(measurement_date) >= 7:
            print(
                'Measurement date includes redundant data '
                '(everything after seconds specification '
                '– the seventh argument – is ignored).'
            )

        sun_vecs_measurement, extras = utils.get_sun_array(
            years,
            months,
            days,
            hours,
            minutes,
            secs,
        )
        trajectories.append(sun_vecs_measurement)
        infos["measurement"] = len(sun_vecs_measurement)
        infos["date_time_ae"].append(extras)

    if shortest_day:
        years = [2021]
        months = [12]
        days = [21]
        hours = list(range(9, 15, stepsize_hours))
        secs = [0]
        sun_vecs_short, extras = utils.get_sun_array(
            years,
            months,
            days,
            hours,
            minutes,
            secs,
        )
        trajectories.append(sun_vecs_short)
        infos["short"] = len(sun_vecs_short)
        infos["date_time_ae"].append(extras)

    if equinox_spring:
        years = [2022]
        months = [3]
        days = [20]
        hours = list(range(7, 17, stepsize_hours))
        secs = [0]
        sun_vecs_autumn, extras = utils.get_sun_array(
            years,
            months,
            days,
            hours,
            minutes,
            secs,
        )
        trajectories.append(sun_vecs_autumn)
        infos["spring"] = len(sun_vecs_autumn)
        infos["date_time_ae"].append(extras)

    if longest_day:
        years = [2022]
        months = [6]
        days = [21]
        hours = list(range(5, 19, stepsize_hours))
        secs = [0]
        sun_vecs_long, extras = utils.get_sun_array(
            years,
            months,
            days,
            hours,
            minutes, secs,
        )
        trajectories.append(sun_vecs_long)
        infos["long"] = len(sun_vecs_long)  # TODO: Tidy up!
        infos["date_time_ae"].append(extras)

    if len(trajectories) == 0:
        raise ValueError("No day submitted to function")
    trajectories = th.cat(trajectories).to(device)

    return trajectories, infos


def generate_sun_array(
        cfg_sun_directions: CfgNode,
        device: th.device,
        train_vec: Optional[torch.Tensor] = None,
        case: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = cfg_sun_directions
    if not case:
        case = cfg.CASE
        print(case)
    assert case is not None

    if case == "random":
        sun_directions, extras = _random_sun_array(cfg.RAND, device)
    elif case == "grid":
        sun_directions, extras = _grid_sun_array(cfg.GRID, device)
    elif case == "vecs":
        sun_directions, extras = _vec_sun_array(cfg.VECS, device)
    elif case == "spheric":
        sun_directions, extras = _spheric_sun_array(
            cfg.SPHERIC, device, train_vec)
    elif case == "season":
        sun_directions, extras = _season_sun_array(device)
    else:
        raise ValueError("unknown `cfg.CASE` in `generate_sun_rays`")
    return sun_directions, extras


def log_dataset(
        writer: Optional[SummaryWriter],
        targets: torch.Tensor,
        prefix: str = '',
) -> torch.Tensor:
    if writer:
        for (i, target) in enumerate(targets):
            writer.add_image(
                f"{prefix}/target_{i}",
                utils.colorize(target),
            )
    return targets
