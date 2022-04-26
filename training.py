import collections
import os
from typing import Callable, cast, List, Optional, Tuple, Union

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from environment import Environment
from heliostat_models import AbstractHeliostat, ParamGroups
from render import Renderer
import utils

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.optim.Optimizer],
    Tuple[torch.Tensor, torch.Tensor],
]
TestLossFn = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
]

LRScheduler = Union[
    th.optim.lr_scheduler._LRScheduler,
    th.optim.lr_scheduler.ReduceLROnPlateau,
]

TrainObjects = collections.namedtuple(
    'TrainObjects',
    [
        'opt',
        'sched',
        'H',
        'ENV',
        'R',
        'targets',
        'sun_directions',
        'loss_func',
        'epoch',
        'prefix',
        'writer',
    ],
    # 'writer' is None by default
    defaults=[None],
)


def _insert_param_group_config(cfg: CfgNode, params: ParamGroups) -> None:
    for param_group in params:
        name = param_group['name']

        if name == 'surface':
            # These parameters are given by the defaults anyway (by
            # definition), so we don't set them explicitly.
            continue

        group_cfg: CfgNode = getattr(cfg, name.upper())

        for (key, value) in group_cfg.items():
            param_group[key.lower()] = value


def _build_optimizer(
        cfg_optimizer: CfgNode,
        params: ParamGroups,
) -> th.optim.Optimizer:
    cfg = cfg_optimizer
    name = cfg.NAME.lower()

    _insert_param_group_config(cfg, params)

    if name == "adam":
        opt: th.optim.Optimizer = th.optim.Adam(
            params,
            lr=cfg.LR,
            betas=(cfg.BETAS[0], cfg.BETAS[1]),
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif name == "adamax":
        opt = th.optim.Adamax(
            params,
            lr=cfg.LR,
            betas=(cfg.BETAS[0], cfg.BETAS[1]),
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif name == "adamw":
        opt = th.optim.AdamW(
            params,
            lr=cfg.LR,
            betas=(cfg.BETAS[0], cfg.BETAS[1]),
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif name == "lbfgs":
        list_params = [param for group in params for param in group['params']]
        opt = th.optim.LBFGS(
            list_params,
            lr=cfg.LR,
        )
    else:
        raise ValueError(
            "Optimizer name not found, change name or implement new optimizer")

    return opt


def _build_scheduler(
        cfg_scheduler: CfgNode,
        opt: th.optim.Optimizer,
        total_steps: int,
) -> LRScheduler:
    name = cfg_scheduler.NAME.lower()
    if name == "reduceonplateau":
        cfg: CfgNode = cfg_scheduler.ROP
        sched: LRScheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=cfg.FACTOR,
            min_lr=cfg.MIN_LR,
            patience=cfg.PATIENCE,
            cooldown=cfg.COOLDOWN,
            verbose=cfg.VERBOSE,
        )
    elif name == "cyclic":
        cfg = cfg_scheduler.CYCLIC
        sched = th.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=cfg.BASE_LR,
            max_lr=cfg.MAX_LR,
            step_size_up=cfg.STEP_SIZE_UP,
            cycle_momentum=cfg.CYCLE_MOMENTUM,
            mode=cfg.MODE,
        )
    elif name == "exponential":
        cfg = cfg_scheduler.EXP
        sched = th.optim.lr_scheduler.ExponentialLR(opt, cfg.GAMMA)
    elif name == "onecycle":
        cfg = cfg_scheduler.ONE_CYCLE
        sched = th.optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined]
            opt,
            total_steps=total_steps,
            max_lr=cfg.MAX_LR,
            pct_start=cfg.PCT_START,
            div_factor=cfg.MAX_LR / cfg.START_LR,
            final_div_factor=cfg.MAX_LR / cfg.FINAL_LR,
            three_phase=cfg.THREE_PHASE,
        )
    else:
        raise ValueError(
            "Scheduler name not found, change name or implement new scheduler")

    return sched


def get_opt_cp_path(cp_path: str) -> str:
    return os.path.expanduser(cp_path[:-3] + '_opt.pt')


def load_optimizer_state(
        opt: th.optim.Optimizer,
        cp_path: str,
        device: th.device,
) -> None:
    cp_path = os.path.expanduser(cp_path)
    if not os.path.isfile(cp_path):
        print(
            f'Warning: cannot find optimizer under {cp_path}; '
            f'please rename your optimizer checkpoint accordingly. '
            f'Continuing with newly created optimizer...'
        )
        return
    cp = th.load(cp_path, map_location=device)
    opt.load_state_dict(cp['opt'])


def build_optimizer_scheduler(
        cfg: CfgNode,
        total_steps: int,
        params: ParamGroups,
        device: th.device,
) -> Tuple[th.optim.Optimizer, LRScheduler]:
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params)
    # Load optimizer state.
    if cfg.LOAD_OPTIMIZER_STATE:
        opt_cp_path = get_opt_cp_path(cfg.CP_PATH)
        load_optimizer_state(opt, opt_cp_path, device)

    sched = _build_scheduler(cfg.TRAIN.SCHEDULER, opt, total_steps)
    return opt, sched


def l1_weight_penalty(
        opt: th.optim.Optimizer,
        param_group_name: Optional[str],
) -> torch.Tensor:
    no_filter = param_group_name is None
    weight_penalty = sum(
        th.linalg.norm(
            th.linalg.norm(
                th.linalg.norm(param, ord=1, dim=-1),
                ord=1,
                dim=-1,
            ),
            ord=1,
            dim=-1
        )
        for group in opt.param_groups
        for param in group['params']
        if no_filter or group['name'] == param_group_name
    )
    assert isinstance(weight_penalty, th.Tensor)
    return weight_penalty


def build_loss_funcs(
        cfg_loss: CfgNode,
        to_optimize: List[str],
) -> Tuple[LossFn, TestLossFn]:
    cfg = cfg_loss
    name = cfg.NAME.lower()
    if name == "mse":
        primitive_loss_func: TestLossFn = th.nn.MSELoss()
    elif name == "l1":
        primitive_loss_func = th.nn.L1Loss()
    else:
        raise ValueError(
            "Loss function name not found, change name or implement new loss")

    def test_loss_func(
            pred_bitmap: torch.Tensor,
            target_bitmap: torch.Tensor,
    ) -> torch.Tensor:
        loss = primitive_loss_func(pred_bitmap, target_bitmap)
        loss /= pred_bitmap.numel()
        return loss

    def loss_func(
            pred_bitmap: torch.Tensor,
            target_bitmap: torch.Tensor,
            opt: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_loss = test_loss_func(pred_bitmap, target_bitmap)
        loss = raw_loss.clone()

        if isinstance(opt, th.optim.LBFGS) and cfg.USE_L1_WEIGHT_DECAY:
            weight_penalty = l1_weight_penalty(opt, None)
            weight_decay_factor: float = cfg.WEIGHT_DECAY_FACTOR
            loss += weight_decay_factor * weight_penalty
        else:
            for name in to_optimize:
                if name == 'surface':
                    node: CfgNode = cfg
                else:
                    node = getattr(cfg, name.upper())

                if not node.USE_L1_WEIGHT_DECAY:
                    continue

                # TODO This is very inefficient as we do a N^2 loop.
                #      However, N is super small so we shouldn't care.
                weight_penalty = l1_weight_penalty(opt, name)
                weight_decay_factor = node.WEIGHT_DECAY_FACTOR
                loss += weight_decay_factor * weight_penalty
        return loss, raw_loss

    return loss_func, test_loss_func


def calc_batch_loss(
        train_objects: TrainObjects,
        return_extras: bool = True,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    # print(epoch)
    # if epoch == 0:
    #     last_lr = opt.param_groups[0]["lr"]
    (
        opt,
        # Don't need scheduler.
        _,
        H,
        ENV,
        R,
        targets,
        sun_directions,
        loss_func,
        epoch,
        prefix,
        writer,
    ) = train_objects
    assert prefix, "prefix string cannot be empty"
    # Initialize Parameters
    # =====================
    loss = th.tensor(0.0, dtype=targets.dtype, device=H.device)
    raw_loss = th.zeros_like(loss)
    if return_extras:
        num_missed = th.tensor(0.0, dtype=targets.dtype, device=H.device)

    # Batch Loop
    # ==========
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        H_aligned = H.align(sun_direction)
        pred_bitmap, (ray_directions, indices, _, _) = R.render(
            H_aligned, return_extras=True)
        # pred_bitmap = pred_bitmap.unsqueeze(0)
        # print(pred_bitmap.shape)
        curr_loss, curr_raw_loss = loss_func(pred_bitmap, target, opt)
        loss += curr_loss / len(targets)
        raw_loss += curr_raw_loss / len(targets)

        with th.no_grad():
            # Plot target images to TensorBoard
            if writer and epoch % 50 == 0:
                writer.add_image(
                    f"{prefix}/prediction_{i}",
                    utils.colorize(pred_bitmap),
                    epoch,
                )

            if return_extras:
                # Compare metrics
                num_missed += (
                    (indices.numel() - indices.count_nonzero())
                    / len(targets)
                )

    if return_extras:
        return loss, (raw_loss, pred_bitmap, num_missed)
    return loss


def calc_batch_grads(
        train_objects: TrainObjects,
        return_extras: bool = True,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    train_objects.opt.zero_grad(set_to_none=True)

    loss, (raw_loss, pred_bitmap, num_missed) = calc_batch_loss(
        train_objects, return_extras=True)

    loss.backward()
    if return_extras:
        return loss, (raw_loss, pred_bitmap, num_missed)
    return loss


def train_batch(
        train_objects: TrainObjects,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    opt = train_objects.opt
    if isinstance(opt, th.optim.LBFGS):
        with th.no_grad():
            _, (raw_loss, pred_bitmap, num_missed) = calc_batch_loss(
                train_objects)
        loss = cast(
            th.Tensor,
            opt.step(cast(
                Callable[[], float],
                lambda: calc_batch_grads(train_objects, return_extras=False),
            )),
        )
    else:
        loss, (raw_loss, pred_bitmap, num_missed) = calc_batch_grads(
            train_objects)
        opt.step()

    # Plot loss to Tensorboard
    with th.no_grad():
        prefix = train_objects.prefix
        assert prefix, "prefix string cannot be empty"
        writer = train_objects.writer
        if writer:
            writer.add_scalar(
                f"{prefix}/loss", loss.item(), train_objects.epoch)
            writer.add_scalar(
                f"{prefix}/raw_loss", raw_loss.item(), train_objects.epoch)

    # Update training parameters
    # ==========================
    sched = train_objects.sched
    if isinstance(sched, th.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(loss)
    else:
        sched.step()

    train_objects.H.step(verbose=True)
    # if opt.param_groups[0]["lr"] < last_lr:
    #     last_lr = opt.param_groups[0]["lr"]

    return loss, raw_loss, pred_bitmap, num_missed


@th.no_grad()
def test_batch(
        heliostat: AbstractHeliostat,
        env: Environment,
        renderer: Renderer,
        targets: torch.Tensor,
        sun_directions: torch.Tensor,
        loss_func: TestLossFn,
        epoch: int,
        prefix: str,
        writer: Optional[SummaryWriter] = None,
        reduction: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert prefix, "prefix string cannot be empty"

    mean_loss = th.tensor(0.0, dtype=targets.dtype, device=heliostat.device)
    losses = []
    bitmaps: Optional[torch.Tensor] = None
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        heliostat_aligned = heliostat.align(sun_direction)
        pred_bitmap = cast(th.Tensor, renderer.render(heliostat_aligned))

        if bitmaps is None:
            bitmaps = th.empty(
                (len(targets),) + pred_bitmap.shape,
                dtype=pred_bitmap.dtype,
            )

        loss = loss_func(pred_bitmap, target)
        if reduction:
            mean_loss += loss / len(targets)
        else:
            losses.append(loss)
        bitmaps[i] = pred_bitmap.detach().cpu()

        if writer:
            writer.add_image(
                f"{prefix}/prediction_{i}", utils.colorize(pred_bitmap), epoch)

    if writer:
        writer.add_scalar(f"{prefix}/loss", mean_loss.item(), epoch)
    assert bitmaps is not None
    if reduction:
        return mean_loss, bitmaps
    else:
        return th.stack(losses), bitmaps
