import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch3d.transforms as throt
import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torchvision as thv
from yacs.config import CfgNode

from environment import Environment
import hausdorff_distance
from heliostat_models import AbstractHeliostat
from render import Renderer
import utils


def _calc_mean_argval(bitmap: torch.Tensor) -> torch.Tensor:
    dtype = bitmap.dtype
    device = bitmap.device

    x_indices = th.arange(bitmap.shape[0], dtype=dtype, device=device)
    y_indices = th.arange(bitmap.shape[1], dtype=dtype, device=device)
    indices = th.cartesian_prod(x_indices, y_indices)

    mean_argval = (indices * bitmap.reshape(-1, 1) / bitmap.mean()).mean(dim=0)
    return mean_argval


def mean_ray_direction(
        bitmap: torch.Tensor,
        position_on_field: torch.Tensor,
        env: Environment,
) -> torch.Tensor:
    assert bitmap.ndim == 2
    mean_argval_x, mean_argval_y = _calc_mean_argval(bitmap)

    # If our bitmap has different values X, we assume index locations so
    # that they are in the center of each bitmap value X in the
    # following diagram. However, we also have values "between" the
    # centers of each X, indicated by a dot (.):
    #
    # .......
    # .X.X.X.
    # .......
    # .X.X.X.
    # .......
    #
    # We normalize the index values accordingly in [0, 1], so that their
    # pointed-to locations match this desired form. In practice, 0 and 1
    # are never reached due to the locations of the X.
    mean_argval_x *= 2
    mean_argval_x += 1
    mean_argval_x /= bitmap.shape[0] * 2
    center_offset_x = (
        env.receiver_plane_x * mean_argval_x
        - env.receiver_plane_x / 2
    )

    mean_argval_y *= 2
    mean_argval_y += 1
    mean_argval_y /= bitmap.shape[1] * 2
    center_offset_y = (
        env.receiver_plane_y * mean_argval_y
        - env.receiver_plane_y / 2
    )

    center_offset = th.stack([
        center_offset_x, center_offset_y, th.zeros_like(center_offset_x)])
    return env.receiver_center + center_offset - position_on_field


def mean_ray_directions(
        bitmaps: torch.Tensor,
        position_on_field: torch.Tensor,
        env: Environment,
) -> torch.Tensor:
    return th.stack([
        mean_ray_direction(bitmap, position_on_field, env)
        for bitmap in bitmaps
    ])


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
                dtype=sun_direction.dtype,
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

    H_aligned = H.align(sun_direction)
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
        prefix: str,
        writer: Optional[SummaryWriter] = None,
) -> torch.Tensor:
    assert prefix, "prefix string cannot be empty"

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
                dtype=sun_directions.dtype,
                device=device,
            )
        targets[i] = target_bitmap
    assert targets is not None
    log_dataset(prefix, writer, targets)
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
        # All interval end values + 1 due to `th.randint` handling the
        # upper limit exclusively.
        year = int(th.randint(1970, 2051, ()).item())
        month = int(th.randint(1, 13, ()).item())
        # Exclude late days in month, just cause a lot of if-clauses.
        day = int(th.randint(1, 30, ()).item())
        hour = int(th.randint(1, 24, ()).item())
        minute = int(th.randint(1, 60, ()).item())
        sec = int(th.randint(1, 60, ()).item())
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
        dtype=azi.dtype,
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
        train_vec: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_vec is None:
        raise ValueError(
            "train_vec is None. Spheric testing needs a train vector")
    if not train_vec.squeeze().shape == (3,):
        raise ValueError(
            "multiple sun vectors detected. "
            "spheric plot is only possible using 1 vector"
        )
    dtype = train_vec.dtype

    train_vec = train_vec / th.linalg.norm(train_vec, dim=1).unsqueeze(1)
    ae = utils.vec_to_ae(train_vec).squeeze()
    ele_angles = th.linspace(
        0.1 * 180,
        0.9 * 180,
        cfg.NUM_SAMPLES,
        dtype=dtype,
        device=device,
    )
    azi_angles = th.linspace(
        0,
        180,
        cfg.NUM_SAMPLES,
        dtype=dtype,
        device=device,
    )

    north = th.tensor([[0, 1, 0]], dtype=dtype, device=device)
    t_ele = throt.Transform3d(dtype=dtype, device=device) \
                 .rotate_axis_angle(ele_angles, "X") \
                 .rotate_axis_angle(-ae[0], "Z")
    spheric_vecs_ele = t_ele.transform_normals(north).squeeze()

    other_direction = train_vec.clone()
    t_azi_western = throt.Transform3d(dtype=dtype, device=device) \
                         .rotate_axis_angle(azi_angles, "Z")
    spheric_vecs_azi_west = t_azi_western.transform_normals(
        other_direction).squeeze()
    t_azi_eastern = throt.Transform3d(dtype=dtype, device=device) \
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
        device: th.device,
        stepsize_hours: int = 1,
        stepsize_minutes: int = 30,
        longest_day: bool = True,
        shortest_day: bool = True,
        equinox_spring: bool = True,
        measurement_day: bool = False,
        *measurement_date: List[int],
        use_old_version: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

    trajectories = []
    secs = [0]
    minutes = [0]
    years = [2022]

    # Shortes Day
    months = [12]
    days = [21]
    if use_old_version:
        hours = [9, 12]
    else:
        hours = [9]
    sun_vecs_short, extras = utils.get_sun_array(
        years,
        months,
        days,
        hours,
        minutes,
        secs,
    )
    trajectories.append(sun_vecs_short)

    # Equinox
    months = [3]
    days = [20]
    if use_old_version:
        hours = [9, 12, 15]
    else:
        hours = [12]
    sun_vecs_autumn, extras = utils.get_sun_array(
        years,
        months,
        days,
        hours,
        minutes,
        secs,
    )
    trajectories.append(sun_vecs_autumn)
    # Longest Day
    months = [6]
    days = [21]
    if use_old_version:
        hours = [9, 12, 15, 18]
    else:
        hours = [18]
    sun_vecs_long, extras = utils.get_sun_array(
        years,
        months,
        days,
        hours,
        minutes,
        secs,
    )
    trajectories.append(sun_vecs_long)

    trajectories = th.cat(trajectories).to(device)
    infos: Dict[str, Any] = {}
    return trajectories, infos


def generate_sun_array(
        cfg_sun_directions: CfgNode,
        device: th.device,
        train_vec: Optional[torch.Tensor] = None,
        case: Optional[str] = None
) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]]:
    cfg = cfg_sun_directions
    if not case:
        case = cfg.CASE
        print(case)
    assert case is not None

    if case == "random":
        extras: Union[torch.Tensor, Dict[str, Any]]
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


def log_contoured(
        prefix: str,
        writer: Optional[SummaryWriter],
        targets: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
) -> torch.Tensor:
    assert prefix, "prefix string cannot be empty"
    if writer:
        contoured_targets = hausdorff_distance.contour_images(
            targets,
            contour_vals,
            contour_val_radius,
        )
        for (i, contoured) in enumerate(contoured_targets):
            writer.add_image(
                f"{prefix}/target_{i}_contoured",
                contoured.unsqueeze(0),
            )
    return contoured_targets


def log_dataset(
        prefix: str,
        writer: Optional[SummaryWriter],
        targets: torch.Tensor,
) -> torch.Tensor:
    assert prefix, "prefix string cannot be empty"
    if writer:
        for (i, target) in enumerate(targets):
            writer.add_image(
                f"{prefix}/target_{i}",
                utils.colorize(target),
            )
    return targets


def read_image(path: str, device: th.device) -> torch.Tensor:
    img = thv.io.read_image(path, thv.io.image.ImageReadMode.GRAY)
    img = img.to(dtype=th.get_default_dtype(), device=device)
    return img


@th.no_grad()
def load_images(
        paths: List[str],
        height: int,
        width: int,
        device: th.device,
        prefix: str,
        writer: Optional[SummaryWriter] = None,
) -> torch.Tensor:
    target_imgs = [read_image(path, device) for path in paths]
    img_transform = thv.transforms.Compose([
        thv.transforms.Resize((height, width)),
        # Remove single channel.
        thv.transforms.Lambda(lambda image: image.squeeze(0)),
        # Try to remove background (i.e. make background dark).
        thv.transforms.Lambda(lambda image: th.clip(
            image - image.mean(),
            min=0,
        )),
        # # Normalize to [0, 1].
        # thv.transforms.Lambda(lambda image: image / image.max()),
        # Normalize intensities, same as renderer.
        thv.transforms.Lambda(lambda image: image / image.sum()),
    ])
    targets = th.stack(list(map(img_transform, target_imgs)))
    log_dataset(prefix, writer, targets)
    return targets
