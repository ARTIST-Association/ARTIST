import collections
import itertools
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from pytorch_minimize.optim import BasinHoppingWrapper, torch_optimizer
import torch
import torch as th
from yacs.config import CfgNode

from environment import Environment
import hausdorff_distance
from heliostat_models import ParamGroups
import plotter
import utils

LossFn = Callable[
    [
        torch.Tensor,  # pred_bitmap
        torch.Tensor,  # target_bitmap
        torch.Tensor,  # z_alignment
        torch.Tensor,  # target_z_alignment
        Optional[torch.Tensor],  # target_set
        torch.Tensor,  # dx_ints
        torch.Tensor,  # dy_ints
        Environment,  # env
        torch.optim.Optimizer,  # opt
    ],
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
        'target_z_alignments',
        'target_sets',
        'sun_directions',
        'loss_func',
        'config',
        'epoch',
        'prefix',
        'writer',
        'test_objects',
        'index',
        'prealignment'
    ],
)

TestObjects = collections.namedtuple(
    'TestObjects',
    [
         'H',
         'ENV',
         'R',
         'test_targets',
         'test_target_sets',
         'test_sun_directions',
         'test_loss_func',
         'cfg',
         'epoch',
         'prefix',
         'writer',
         'H_target',
         'logdir',
         'reduction',
         'prealignment',
    ],
)


def normalize_param_group_name(name: str) -> Tuple[str, str]:
    if not name[-1].isdigit():
        return name, ''
    underscore_index = name.rfind('_')
    if underscore_index > 0 and name[underscore_index + 1:].isdigit():
        return name[:underscore_index], name[:underscore_index]
    else:
        return name, ''


def _insert_param_group_config(cfg: CfgNode, params: ParamGroups) -> None:
    for param_group in params:
        name = param_group['name']
        normalized_name, _ = normalize_param_group_name(name)

        if name == 'surface':
            # These parameters are given by the defaults anyway (by
            # definition), so we don't set them explicitly.
            continue
        elif normalized_name == 'surface':
            # Assume we use a NURBS surface.
            # Set LR lower for less influential parameters.
            base_lr: float = cfg.LR
            # TODO Do we also need to take spline degree into account?
            param_group['lr'] = base_lr
            # Was worse when we used location-dependent LRs.
            # param_group['lr'] = base_lr * {
            #     'surface_0': 1.0,  # inner
            #     'surface_1': 0.75,  # edges
            #     'surface_2': 0.5,  # corners
            # }[name]
            continue

        group_cfg: CfgNode = getattr(cfg, normalized_name.upper())

        for (key, value) in group_cfg.items():
            param_group[key.lower()] = value


def _build_optimizer(
        cfg_optimizer: CfgNode,
        params: ParamGroups,
        total_steps: Optional[int] = None,
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
    elif name == 'basinhopping':
        cfg_global: CfgNode = cfg.GLOBAL
        list_params = [param for group in params for param in group['params']]

        minimizer_config: Dict[str, Any] = {'jac': True}

        inner_niter: int = cfg_global.NUM_MIN_ITER

        minimizer_name: str = cfg_global.USE_MINIMIZER
        if minimizer_name.lower() == "adam":
            optim_cls = th.optim.Adam
            optim_kwargs = dict(
                lr=cfg.LR,
                betas=(cfg.BETAS[0], cfg.BETAS[1]),
                eps=cfg.EPS,
                weight_decay=cfg.WEIGHT_DECAY,
            )

            use_sched: bool = cfg_global.USE_SCHEDULER
            sched_cls = th.optim.lr_scheduler.ReduceLROnPlateau
            sched_kwargs = dict(
                factor=cfg.FACTOR,
                min_lr=cfg.MIN_LR,
                patience=cfg.PATIENCE,
                cooldown=cfg.COOLDOWN,
                verbose=cfg.VERBOSE,
            )

            minimizer_config['method'] = torch_optimizer
            minimizer_config['options'] = {
                'optim_cls': optim_cls,
                'optim_kwargs': optim_kwargs,
                'disp': True,
                'niter': inner_niter,
            }
            if use_sched:
                minimizer_config['options']['sched_cls'] = sched_cls
                minimizer_config['options']['sched_kwargs'] = sched_kwargs
        else:
            minimizer_config['method'] = minimizer_name
            minimizer_config['options'] = {
                'disp': True,
                'niter': inner_niter,
                'maxiter': inner_niter,
                # 'gtol': 1e-4,
            }

        outer_niter: int = cfg_global.NUM_BASIN_ITER
        temperature: float = cfg_global.TEMP
        stepsize: float = cfg_global.STEP_SIZE
        basinhopping_config = {
            'disp': True,
            'niter': outer_niter,
            'T': temperature,
            'stepsize': stepsize,
        }
        opt = BasinHoppingWrapper(
            list_params,
            minimizer_config,
            basinhopping_config,
        )
    elif name == 'adan':
        try:
            from adan import Adan
        except ImportError:
            raise ImportError(
                'to use the Adan optimizer, please execute '
                '`python3 -m pip install '
                'git+https://github.com/janEbert/Adan.git`'
            )
        beta_3 = cfg.BETAS[2] if len(cfg.BETAS) > 2 else 0.99
        opt = Adan(
            params,
            lr=cfg.LR,
            betas=(cfg.BETAS[0], cfg.BETAS[1], beta_3),
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
            foreach=True,
        )
    elif name == 'velo':
        assert total_steps is not None, \
            'VeLO optimizer needs a number of training steps'
        try:
            from pytorch_velo import VeLO
        except ImportError:
            raise ImportError(
                'to use the VeLO optimizer, please execute '
                '`python3 -m pip install '
                'git+https://github.com/janEbert/PyTorch-VeLO.git`'
            )
        opt = VeLO(
            params,
            num_training_steps=total_steps,
            weight_decay=cfg.WEIGHT_DECAY,
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
    if isinstance(opt, BasinHoppingWrapper):
        class DummySched(th.optim.lr_scheduler._LRScheduler):
            def __init__(self) -> None:
                pass

            def step(self, epoch: Optional[int] = None) -> None:
                pass

        sched: LRScheduler = DummySched()
    elif name == "reduceonplateau":
        cfg: CfgNode = cfg_scheduler.ROP
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
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
            # Handle zero steps (e.g. for no pretraining).
            total_steps=max(total_steps, 1),
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
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params, total_steps)
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


def _get_loss_func(
        cfg_loss: CfgNode,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = cfg_loss.NAME.lower()
    if name == "mse":
        primitive_loss_func: TestLossFn = th.nn.MSELoss()
    elif name == "l1":
        primitive_loss_func = th.nn.L1Loss()
    else:
        raise ValueError(
            "Loss function name not found, change name or implement new loss")
    return primitive_loss_func


def if_else_zero(
        condition: bool,
        closure: Callable[[], torch.Tensor],
        dtype: th.dtype,
        device: th.device,
) -> torch.Tensor:
    if condition:
        result = closure()
    else:
        result = th.tensor(0.0, dtype=dtype, device=device)
    return result


def build_loss_funcs(
        cfg_loss: CfgNode,
        to_optimize: List[str],
) -> Tuple[LossFn, TestLossFn]:
    cfg = cfg_loss
    primitive_loss_func = _get_loss_func(cfg)
    miss_primitive_loss_func = _get_loss_func(cfg.MISS)
    alignment_primitive_loss_func = _get_loss_func(cfg.ALIGNMENT)

    loss_factor: float = cfg.FACTOR
    miss_loss_factor: float = cfg.MISS.FACTOR
    alignment_loss_factor: float = cfg.ALIGNMENT.FACTOR
    hausdorff_loss_factor: float = cfg.HAUSDORFF.FACTOR

    def test_loss_func(
            pred_bitmap: torch.Tensor,
            target_bitmap: torch.Tensor,
    ) -> torch.Tensor:
        loss = primitive_loss_func(pred_bitmap, target_bitmap)
        loss *= loss_factor
        return loss

    def miss_loss_func(
            pred_bitmap: torch.Tensor,
            dx_ints: torch.Tensor,
            dy_ints: torch.Tensor,
            env: Environment,
    ) -> torch.Tensor:
        miss_loss = miss_primitive_loss_func(
            dx_ints,
            th.clip(dx_ints, min=-1, max=env.receiver_plane_x + 1),
        ) * miss_loss_factor
        miss_loss += miss_primitive_loss_func(
            dy_ints,
            th.clip(dy_ints, min=-1, max=env.receiver_plane_y + 1),
        ) * miss_loss_factor
        miss_loss /= len(dx_ints) * len(dy_ints)
        return miss_loss

    def hausdorff_loss_func(
            pred_bitmap: torch.Tensor,
            target_set: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert target_set is not None
        weighted_hausdorff_dists = \
            hausdorff_distance.weighted_hausdorff_distance(
                pred_bitmap.unsqueeze(0),
                [target_set],
                norm_p=cfg.HAUSDORFF.NORM_P,
                mean_p=cfg.HAUSDORFF.MEAN_P,
            )
        weighted_hausdorff_dists[th.isposinf(weighted_hausdorff_dists)] = \
            10 * hausdorff_distance.max_hausdorff_distance(
                pred_bitmap.shape,
                pred_bitmap.device,
                pred_bitmap.dtype,
                norm_p=cfg.HAUSDORFF.NORM_P,
            )
        hausdorff_loss = (
            weighted_hausdorff_dists.mean()
            * hausdorff_loss_factor
        )
        return hausdorff_loss

    def loss_func(
            pred_bitmap: torch.Tensor,
            target_bitmap: torch.Tensor,
            z_alignment: torch.Tensor,
            target_z_alignment: torch.Tensor,
            target_set: Optional[torch.Tensor],
            dx_ints: torch.Tensor,
            dy_ints: torch.Tensor,
            env: Environment,
            opt: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = pred_bitmap.dtype
        device = pred_bitmap.device

        raw_loss = if_else_zero(
            loss_factor != 0,
            lambda: test_loss_func(
                pred_bitmap,
                target_bitmap,
            ),
            dtype,
            device,
        )
        loss = raw_loss.clone()

        # Penalize misses
        miss_loss = if_else_zero(
            miss_loss_factor != 0,
            lambda: miss_loss_func(pred_bitmap, dx_ints, dy_ints, env),
            dtype,
            device,
        )
        loss += miss_loss

        # Penalize misalignment
        # TODO Does this even make sense when using active canting?
        alignment_loss = if_else_zero(
            alignment_loss_factor != 0,
            lambda: alignment_primitive_loss_func(
                (
                    z_alignment.mean(dim=0)
                    if z_alignment.ndim > 1 and target_z_alignment.ndim == 1
                    else z_alignment
                ),
                (
                    target_z_alignment.mean(dim=0)
                    if target_z_alignment.ndim > 1 and z_alignment.ndim == 1
                    else target_z_alignment
                ),
            ) * alignment_loss_factor,
            dtype,
            device,
        )
        loss += alignment_loss

        # Weighted Hausdorff loss
        hausdorff_loss = if_else_zero(
            hausdorff_loss_factor != 0,
            lambda: hausdorff_loss_func(pred_bitmap, target_set),
            dtype,
            device,
        )
        loss += hausdorff_loss

        if (
                isinstance(opt, (th.optim.LBFGS, BasinHoppingWrapper))
                and cfg.USE_L1_WEIGHT_DECAY
        ):
            weight_penalty = l1_weight_penalty(opt, None)
            weight_decay_factor: float = cfg.WEIGHT_DECAY_FACTOR
            loss += weight_decay_factor * weight_penalty
        else:
            for name in to_optimize:
                normalized_name, _ = normalize_param_group_name(name)

                if normalized_name == 'surface':
                    node: CfgNode = cfg
                else:
                    node = getattr(cfg, normalized_name.upper())

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
        minimizer_epoch: Optional[int] = None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    (
        opt,
        # Don't need scheduler.
        _,
        H,
        ENV,
        R,
        targets,
        target_z_alignments,
        target_sets,
        sun_directions,
        loss_func,
        config,
        epoch,
        prefix,
        writer,
        test_objects,
        index,
        prealignments,
    ) = train_objects
    assert prefix, "prefix string cannot be empty"

    # print(index)
    if minimizer_epoch:
        epoch = minimizer_epoch

    # Initialize Parameters
    # =====================
    # print(targets.shape,target_z_alignments.shape,sun_directions.shape)
    loss = th.tensor(0.0, dtype=targets.dtype, device=H.device)
    raw_loss = th.zeros_like(loss)
    if return_extras:
        num_missed = th.tensor(0.0, dtype=targets.dtype, device=H.device)

    # Batch Loop
    # ==========
    for (i, (
            target,
            target_z_alignment,
            target_set,
            sun_direction,
            prealignment,
    )) in enumerate(zip(
            targets,
            target_z_alignments,
            target_sets if target_sets is not None else itertools.repeat(None),
            sun_directions,
            (
                prealignments
                if prealignments is not None
                else itertools.repeat(None)
            ),
    )):
        if index is None:
            index = i
        if prealignment:
            H.disturbance_angles = prealignment
        H_aligned = H.align(sun_direction)
        (
            pred_bitmap,
            (ray_directions, dx_ints, dy_ints, indices, _, _),
        ) = R.render(H_aligned, return_extras=True)
        # pred_bitmap = pred_bitmap.unsqueeze(0)
        # print(th.sum(pred_bitmap), th.sum(target))
        curr_loss, curr_raw_loss = loss_func(
            pred_bitmap,
            target,
            H_aligned.alignment[..., -1, :],
            target_z_alignment,
            target_set,
            dx_ints,
            dy_ints,
            ENV,
            opt,
        )
        utils.to_tensorboard(
            writer,
            prefix,
            epoch,
            image=pred_bitmap,
            plot_interval=config.TRAIN.IMG_INTERVAL,
            index=i,
        )

        loss += curr_loss / len(targets)
        raw_loss += curr_raw_loss / len(targets)

        with torch.no_grad():
            if return_extras:
                # Compare metrics
                num_missed += (
                    (indices.numel() - indices.count_nonzero())
                    / len(targets)
                )
    # exit()
    if return_extras:
        return loss, (raw_loss, pred_bitmap, num_missed)
    return loss


def calc_batch_grads(
        train_objects: TrainObjects,
        return_extras: bool = True,
        minimizer_epoch: Optional[int] = None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    train_objects.opt.zero_grad(set_to_none=True)

    loss, (raw_loss, pred_bitmap, num_missed) = calc_batch_loss(
        train_objects, return_extras=True, minimizer_epoch=minimizer_epoch)

    # m = th.nn.Threshold(0.05, 0)
    # pred_bitmap = m(pred_bitmap)
    # plt.imshow(
    #     abs(pred_bitmap - train_objects.targets.squeeze()), cmap="jet")
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)  # TODO ENTFERNEN vor Release
    # ax1.imshow(pred_bitmap.detach().numpy(), cmap="jet")
    # ax1.xaxis.set_major_locator(plt.NullLocator())
    # ax1.yaxis.set_major_locator(plt.NullLocator())
    # ax2.imshow(train_objects.targets.squeeze(), cmap="jet")
    # ax2.set_axis_off()
    # plt.subplots_adjust(
    #     top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
    # print(raw_loss)
    # # plt.savefig("ours.png", bbox_inches='tight', pad_inches=0)
    # exit()
    loss.backward()
    if return_extras:
        return loss, (raw_loss, pred_bitmap, num_missed)
    return loss


def train_batch(
        train_objects: TrainObjects,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    opt = train_objects.opt
    if isinstance(opt, (th.optim.LBFGS, BasinHoppingWrapper)):
        with th.no_grad():
            _, (raw_loss, pred_bitmap, num_missed) = calc_batch_loss(
                train_objects)
        # TODO Minimizer epoch is broken. It will add temporally
        #      non-sensical values in TensorBoard. This needs to be
        #      adjusted according to the global epoch.
        epoch_minimizer = [0]
        best_result = [th.tensor(float('inf'))]

        def global_opt() -> torch.Tensor:
            loss, (raw_loss, pred_bitmap, num_missed) = calc_batch_grads(
                train_objects, minimizer_epoch=epoch_minimizer[0])
            writer = train_objects.writer
            if writer:
                writer.add_scalar(
                    "minimizer/loss", raw_loss.item(), epoch_minimizer[0])
            print(f"Minimizer epoch {epoch_minimizer}. Loss={raw_loss.item()}")

            cfg = train_objects.config
            test_objects = train_objects.test_objects
            if epoch_minimizer[0] % cfg.TEST.INTERVAL == 0:
                test_loss, hausdorff_dist, _ = test_batch(
                    test_objects, epoch_minimizer[0])
                plotter.plot_surfaces_mrad(
                    test_objects.H_target,
                    test_objects.H,
                    epoch_minimizer[0],
                    test_objects.logdir,
                    writer,
                )
                print(
                    f"Minimizer epoch {epoch_minimizer}. "
                    f"Test loss={test_loss.item()}"
                )
                if (
                        test_loss.detach().cpu() < best_result[0]
                        and cfg.SAVE_RESULTS
                ):
                    print(
                        f"New best test loss found. "
                        f"Value: {test_loss.detach().cpu()}. "
                        f"Old Value:{best_result[0]}"
                    )

                    best_result[0] = test_loss.detach().cpu()
            epoch_minimizer[0] += 1

            return loss

        loss = cast(
            th.Tensor,
            opt.step(cast(
                Callable[[], float],
                global_opt,
            )),
        )
    else:
        loss, (raw_loss, pred_bitmap, num_missed) = calc_batch_grads(
            train_objects)
        opt.step()

    # Plot loss to Tensorboard
    # with th.no_grad():
        # prefix = train_objects.prefix
        # assert prefix, "prefix string cannot be empty"
        # writer = train_objects.writer
        # if writer:
        #     # print(writer, prefix, loss.item(),train_objects.epoch)
        #     writer.add_scalar(
        #         f"{prefix}/loss", loss.item(), train_objects.epoch)
        #     writer.add_scalar(
        #         f"{prefix}/raw_loss", raw_loss.item(), train_objects.epoch)

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
        test_objects: TestObjects,
        minimizer_epoch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        heliostat,
        env,
        renderer,
        targets,
        target_sets,
        sun_directions,
        loss_func,
        config,
        epoch,
        prefix,
        writer,
        H_target,
        logdir,
        reduction,
        prealignments,
    ) = test_objects
    assert prefix, "prefix string cannot be empty"

    if minimizer_epoch:
        epoch = minimizer_epoch
        plot_interval: int = config.TEST.INNER_IMG_INTERVAL
    else:
        plot_interval = config.TEST.IMG_INTERVAL
    # print(prealignments)

    mean_loss = th.tensor(0.0, dtype=targets.dtype, device=heliostat.device)
    losses = []
    bitmaps: Optional[torch.Tensor] = None
    for (i, (target, sun_direction, prealignment)) in enumerate(zip(
            targets,
            sun_directions,
            (
                prealignments
                if prealignments is not None
                else itertools.repeat(None)
            ),
    )):
        if prealignment:
            # print(prealignment)
            heliostat.disturbance_angles = prealignment
        heliostat_aligned = heliostat.align(sun_direction)
        (
            pred_bitmap,
            (ray_directions, dx_ints, dy_ints, _, _, _),
        ) = renderer.render(heliostat_aligned, return_extras=True)

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
            utils.to_tensorboard(
                writer,
                prefix,
                epoch,
                image=pred_bitmap,
                plot_interval=plot_interval,
                index=i,
            )
            # writer.add_image(
            #     f"{prefix}/prediction_{i}", pred_bitmap, epoch)
    assert bitmaps is not None

    hausdorff_dists = hausdorff_distance.set_hausdorff_distance(
        hausdorff_distance.images_to_sets(
            bitmaps,
            config.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
            config.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
        ),
        target_sets,
    )
    mean_hausdorff_dist = hausdorff_dists.mean()

    if writer:
        # writer.add_scalar(f"{prefix}/loss", mean_loss.item(), epoch)
        writer.add_scalar(
            f"{prefix}/hausdorff_distance", mean_hausdorff_dist.item(), epoch)
    if reduction:
        return mean_loss, mean_hausdorff_dist, bitmaps
    else:
        return th.stack(losses), hausdorff_dists, bitmaps
