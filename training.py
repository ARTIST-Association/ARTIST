import collections
import os
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from environment import Environment
from heliostat_models import AbstractHeliostat
from render import Renderer
import utils

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.optim.Optimizer],
    torch.Tensor,
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


def _build_optimizer(
        cfg_optimizer: CfgNode,
        params: List[torch.Tensor],
) -> th.optim.Optimizer:
    cfg = cfg_optimizer
    name = cfg.NAME.lower()

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
        opt = th.optim.LBFGS(
            params,
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
        params: List[torch.Tensor],
        device: th.device,
) -> Tuple[th.optim.Optimizer, LRScheduler]:
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params)
    # Load optimizer state.
    if cfg.LOAD_OPTIMIZER_STATE:
        opt_cp_path = get_opt_cp_path(cfg.CP_PATH)
        load_optimizer_state(opt, opt_cp_path, device)

    sched = _build_scheduler(cfg.TRAIN.SCHEDULER, opt, total_steps)
    return opt, sched


def build_loss_funcs(cfg_loss: CfgNode) -> Tuple[LossFn, TestLossFn]:
    cfg = cfg_loss
    name = cfg.NAME.lower()
    if name == "mse":
        test_loss_func: TestLossFn = th.nn.MSELoss()
    elif name == "l1":
        test_loss_func = th.nn.L1Loss()
    else:
        raise ValueError(
            "Loss function name not found, change name or implement new loss")

    def loss_func(
            pred_bitmap: torch.Tensor,
            target_bitmap: torch.Tensor,
            opt: torch.optim.Optimizer,
    ) -> torch.Tensor:
        loss = test_loss_func(pred_bitmap, target_bitmap)

        if cfg.USE_L1_WEIGHT_DECAY:
            weight_decay = sum(
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
            )
            loss += cfg.WEIGHT_DECAY_FACTOR * weight_decay
        return loss

    return loss_func, test_loss_func


def calc_batch_loss(
        train_objects: TrainObjects,
        return_extras: bool = True,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
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
        loss += loss_func(pred_bitmap, target, opt) / len(targets)

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
        return loss, (pred_bitmap, num_missed)
    return loss


def calc_batch_grads(
        train_objects: TrainObjects,
        return_extras: bool = True,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
]:
    train_objects.opt.zero_grad(set_to_none=True)

    loss, (pred_bitmap, num_missed) = calc_batch_loss(
        train_objects, return_extras=True)

    loss.backward()
    if return_extras:
        return loss, (pred_bitmap, num_missed)
    return loss


def train_batch(
        train_objects: TrainObjects,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    opt = train_objects.opt
    if isinstance(opt, th.optim.LBFGS):
        with th.no_grad():
            _, (pred_bitmap, num_missed) = calc_batch_loss(
                train_objects)
        loss: torch.Tensor = opt.step(  # type: ignore[assignment]
            lambda: calc_batch_grads(  # type: ignore[arg-type,return-value]
                train_objects, return_extras=False),
        )
    else:
        loss, (pred_bitmap, num_missed) = calc_batch_grads(
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

    return loss, pred_bitmap, num_missed


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
        pred_bitmap: torch.Tensor = \
            renderer.render(heliostat_aligned)  # type: ignore[assignment]

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
