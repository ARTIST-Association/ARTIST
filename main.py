import collections
import copy
from datetime import datetime
import functools
import os
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

import data
from defaults import get_cfg_defaults, load_config_file
import disk_cache
from environment import Environment
from heliostat_models import AbstractHeliostat, Heliostat
from multi_nurbs_heliostat import MultiNURBSHeliostat
from nurbs_heliostat import AbstractNURBSHeliostat, NURBSHeliostat
import plotter
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


def check_consistency(cfg: CfgNode) -> None:
    print("Loaded Switches:")
    print(f"Heliostat shape: {cfg.H.SHAPE}")
    print(f"Solar distribution: {cfg.AC.SUN.DISTRIBUTION}")
    print(f"Scheduler: {cfg.TRAIN.SCHEDULER.NAME}")
    print(f"Optimizer: {cfg.TRAIN.OPTIMIZER.NAME}")
    print(f"Loss: {cfg.TRAIN.LOSS.NAME}")

    warnings_found = False
    if cfg.TRAIN.LOSS.USE_L1_WEIGHT_DECAY:
        if not cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY == 0:
            warnings_found = True
            print("WARNING: Do you really want to use L2 and L1 weight decay?")
    if cfg.TRAIN.SCHEDULER.NAME.lower() == "cyclic":
        if not cfg.TRAIN.SCHEDULER.CYCLIC.BASE_LR == cfg.TRAIN.OPTIMIZER.LR:
            warnings_found = True
            print(
                "WARNING: Cyclic base LR and optimizer LR should be the same")
    if not cfg.CP_PATH == "":
        print("continue without loading...")
        if not os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
            warnings_found = True
            print(
                "WARNING: Checkpoint path not found; "
                "continuing without loading..."
            )
    if (
            cfg.LOAD_OPTIMIZER_STATE
            and not os.path.isfile(_get_opt_cp_path(cfg.CP_PATH))
    ):
        warnings_found = True
        print(
            "WARNING: Optimizer checkpoint not found; "
            "continuing without loading..."
        )
    if not warnings_found:
        print("No warnings found. Good Luck!")
        print("=============================")


def set_up_dataset_caching(
        device: th.device,
        writer: Optional[SummaryWriter],
) -> Tuple[
    Tuple[Callable, Callable, Callable, Callable],
    Tuple[Callable, Callable, Callable, Callable, Callable, Callable],
]:
    def make_cached_generate_sun_array(prefix=''):
        return disk_cache.disk_cache(
            data.generate_sun_array,
            device,
            'cached',
            prefix,
            ignore_argnums=[1],
        )

    def make_cached_generate_dataset(prefix, tb_log=True):
        if tb_log:
            log_dataset = functools.partial(
                data.log_dataset,
                prefix,
                writer,
            )
        else:
            log_dataset = None

        return disk_cache.disk_cache(
            data.generate_dataset,
            device,
            'cached',
            prefix,
            on_load=log_dataset,
            ignore_argnums=[3, 4, 5],
        )

    return (
        (
            make_cached_generate_sun_array('target_'),
            make_cached_generate_sun_array('test_'),
            make_cached_generate_sun_array('grid_'),
            make_cached_generate_sun_array('spheric_'),
            make_cached_generate_sun_array('season_'),
        ),
        (
            make_cached_generate_dataset('train'),
            make_cached_generate_dataset('pretrain'),
            make_cached_generate_dataset('test'),
            make_cached_generate_dataset('grid', False),
            make_cached_generate_dataset('naive_grid', False),
            make_cached_generate_dataset('spheric', False),
            make_cached_generate_dataset('naive_spheric', False),
            make_cached_generate_dataset('season', False),
            make_cached_generate_dataset('naive_season', False),
        ),
    )


def load_heliostat(cfg: CfgNode, device: th.device) -> AbstractHeliostat:
    cp_path = os.path.expanduser(cfg.CP_PATH)
    cp = th.load(cp_path, map_location=device)
    if cfg.USE_NURBS:
        if 'facets' in cp:
            nurbs_heliostat_cls: Type[AbstractNURBSHeliostat] = \
                MultiNURBSHeliostat
            kwargs = {'receiver_center': cfg.AC.RECEIVER.CENTER}
        else:
            nurbs_heliostat_cls = NURBSHeliostat
            kwargs = {}
        H: AbstractHeliostat = nurbs_heliostat_cls.from_dict(
            cp,
            device,
            nurbs_config=cfg.NURBS,
            config=cfg.H,
            **kwargs,
        )
    else:
        H = Heliostat.from_dict(cp, device)
    return H


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


def _build_multi_nurbs_target(
        cfg: CfgNode,
        device: th.device,
) -> MultiNURBSHeliostat:
    mnh_cfg = cfg.clone()
    mnh_cfg.defrost()
    mnh_cfg.H.SHAPE = 'Ideal'
    mnh_cfg.freeze()

    nurbs_cfg = mnh_cfg.NURBS.clone()
    nurbs_cfg.defrost()

    # We need this to get correct shapes.
    nurbs_cfg.SET_UP_WITH_KNOWLEDGE = True
    # Deactivate good-for-training options.
    nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE = False
    nurbs_cfg.RECALCULATE_EVAL_POINTS = False
    nurbs_cfg.GROWING.INTERVAL = 0
    # Can't use active canting with standard heliostat model.
    nurbs_cfg.FACETS.CANTING.ACTIVE = False

    # Overwrite all attributes specified via `mnh_cfg.H.NURBS`.
    node_stack = [(nurbs_cfg, mnh_cfg.H.NURBS)]
    while node_stack:
        node, h_node = node_stack.pop()

        for attr in node.keys():
            if not hasattr(h_node, attr):
                continue

            if isinstance(getattr(node, attr), CfgNode):
                node_stack.append((
                    getattr(node, attr),
                    getattr(h_node, attr),
                ))
            else:
                setattr(node, attr, getattr(h_node, attr))

    nurbs_cfg.freeze()
    mnh = MultiNURBSHeliostat(
        mnh_cfg.H,
        nurbs_cfg,
        device,
        receiver_center=mnh_cfg.AC.RECEIVER.CENTER,
    )

    for facet in mnh.facets:
        facet.set_ctrl_points(
            facet.ctrl_points
            + th.rand_like(facet.ctrl_points)
            * mnh_cfg.H.NURBS.MAX_ABS_NOISE
        )

    return mnh


def _multi_nurbs_to_standard(
        cfg: CfgNode,
        mnh: MultiNURBSHeliostat,
) -> Heliostat:
    H = Heliostat(cfg.H, mnh.device)
    discrete_points, normals = mnh.discrete_points_and_normals()
    H._discrete_points = discrete_points
    H._normals = normals
    H._normals_ideal = th.cat([
        facet._normals_ideal
        for facet in mnh.facets
    ])
    H.params = mnh.nurbs_cfg
    H.height = mnh.height
    H.width = mnh.width
    H.rows = mnh.rows
    H.cols = mnh.cols

    return H


def build_target_heliostat(cfg: CfgNode, device: th.device) -> Heliostat:
    if cfg.H.SHAPE.lower() == 'nurbs':
        mnh = _build_multi_nurbs_target(cfg, device)
        H = _multi_nurbs_to_standard(cfg, mnh)
    else:
        H = Heliostat(cfg.H, device)
    return H


def build_heliostat(
        cfg: CfgNode,
        device: th.device,
) -> AbstractHeliostat:
    if cfg.CP_PATH and os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
        H = load_heliostat(cfg, device)
    else:
        if cfg.USE_NURBS:
            if (
                    cfg.NURBS.FACETS.POSITIONS is not None
                    and len(cfg.NURBS.FACETS.POSITIONS) > 1
            ):
                nurbs_heliostat_cls: Type[AbstractNURBSHeliostat] = \
                    MultiNURBSHeliostat
                kwargs = {'receiver_center': cfg.AC.RECEIVER.CENTER}
            else:
                nurbs_heliostat_cls = NURBSHeliostat
                kwargs = {}

            H = nurbs_heliostat_cls(cfg.H, cfg.NURBS, device, **kwargs)
        else:
            H = Heliostat(cfg.H, device)
    return H


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


def _get_opt_cp_path(cp_path: str) -> str:
    return os.path.expanduser(cp_path[:-3] + '_opt.pt')


def build_optimizer_scheduler(
        cfg: CfgNode,
        total_steps: int,
        params: List[torch.Tensor],
        device: th.device,
) -> Tuple[th.optim.Optimizer, LRScheduler]:
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params)
    # Load optimizer state.
    if cfg.LOAD_OPTIMIZER_STATE:
        opt_cp_path = _get_opt_cp_path(cfg.CP_PATH)
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

    def loss_func(pred_bitmap, target_bitmap, opt):
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
    loss = th.tensor(0.0, device=H.device)
    if return_extras:
        num_missed = th.tensor(0.0, device=H.device)

    # Batch Loop
    # ==========
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        H_aligned = H.align(sun_direction, ENV.receiver_center)
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
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    mean_loss = th.tensor(0.0, device=heliostat.device)
    losses = []
    bitmaps: Optional[torch.Tensor] = None
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        heliostat_aligned = heliostat.align(
            sun_direction, env.receiver_center)
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


def main(config_file_name: Optional[str] = None) -> None:
    # Load Defaults
    # =============
    # config_file_name = None
    cfg_default = get_cfg_defaults()
    if config_file_name:
        print(f"load: {config_file_name}")
        # config_file = os.path.join("configs", config_file_name)
        cfg = load_config_file(cfg_default, config_file_name)
    else:
        print("No config loaded. Use defaults")
        cfg = cfg_default
    cfg.freeze()

    if cfg.USE_FLOAT64:
        th.set_default_dtype(th.float64)
    else:
        th.set_default_dtype(th.float32)

    # Set up Logging
    # ==============
    if cfg.SAVE_RESULTS:
        now = datetime.now()
        time_str = now.strftime("%y%m%d_%H%M")
        root_logdir = os.path.join(cfg.LOGDIR, cfg.ID)
        logdir: Optional[str] = os.path.join(
            root_logdir,
            cfg.EXPERIMENT_NAME + f"_{time_str}",
        )
        assert logdir is not None
        logdir_files: Optional[str] = os.path.join(logdir, "Logfiles")
        assert logdir_files is not None
        logdir_images: Optional[str] = os.path.join(logdir, "Images")
        assert logdir_images is not None
        logdir_enhanced_test = os.path.join(logdir_images, "EnhancedTest")
        logdir_surfaces = os.path.join(logdir_images, "Surfaces")
        logdir_pretrain_surfaces = os.path.join(logdir_images, "PreSurfaces")
        cfg.merge_from_list(["LOGDIR", logdir])
        os.makedirs(root_logdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(logdir_files, exist_ok=True)
        os.makedirs(logdir_images, exist_ok=True)
        os.makedirs(logdir_enhanced_test, exist_ok=True)

        with open(os.path.join(logdir, "config.yaml"), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)

        writer: Optional[SummaryWriter] = SummaryWriter(logdir)
    else:
        writer = None
        logdir = None
        logdir_files = None
        logdir_images = None

    check_consistency(cfg)
    # Set system params
    # =================
    th.manual_seed(cfg.SEED)
    device = th.device(
        'cuda'
        if cfg.USE_GPU and th.cuda.is_available()
        else 'cpu'
    )

    (
        (
            cached_generate_sun_array,
            cached_generate_test_sun_array,
            cached_generate_grid_sun_array,
            cached_generate_spheric_sun_array,
            cached_generate_season_sun_array
        ),
        (
            cached_generate_target_dataset,
            cached_generate_pretrain_dataset,
            cached_generate_test_dataset,
            cached_generate_grid_dataset,
            cached_generate_naive_grid_dataset,
            cached_generate_spheric_dataset,
            cached_generate_naive_spheric_dataset,
            cached_generate_season_dataset,
            cached_generate_naive_season_dataset,
        ),
    ) = set_up_dataset_caching(device, writer)

    # Create Dataset
    # ==============
    # Create Heliostat Object and Load Model defined in config file
    print("Create dataset using:")
    print(f"Aimpoint: {cfg.AC.RECEIVER.CENTER}")
    print(
        f"Receiver Resolution: {cfg.AC.RECEIVER.RESOLUTION_X}×"
        f"{cfg.AC.RECEIVER.RESOLUTION_Y}"
    )
    print("=============================")
    H_target = build_target_heliostat(cfg, device)
    ENV = Environment(cfg.AC, device)
    sun_directions, ae = cached_generate_sun_array(
        cfg.TRAIN.SUN_DIRECTIONS, device)

    targets = cached_generate_target_dataset(
        H_target,
        ENV,
        sun_directions,
        logdir_files,
        "train",
        writer,
    )
    # H_naive_target = build_target_heliostat(cfg, device)
    # H_naive_target._normals = H_naive_target._normals_ideal
    # naive_targets = cached_generate_pretrain_dataset(
    #     H_naive_target,
    #     ENV,
    #     sun_directions,
    #     logdir_files,
    #     writer,
    #     "pretrain"
    # )

    test_sun_directions, test_ae = cached_generate_test_sun_array(
        cfg.TEST.SUN_DIRECTIONS, device)

    test_targets = cached_generate_test_dataset(
        H_target,
        ENV,
        test_sun_directions,
        None,
        "test",
        writer,
    )
    # plotter.plot_surfaces_mm(H_target, H_target, 1, logdir)

    # Better Testing
    # ==============
    # state = th.random.get_rng_state()

    # if cfg.TEST.PLOT.GRID == True or cfg.TEST.PLOT.SPHERIC == True or cfg.TEST.PLOT.SEASON:
    #     H_validation = build_target_heliostat(cfg, device)
    #     ENV_validation = Environment(cfg.AC, device)
    # if cfg.TEST.PLOT.GRID:
    #     (
    #         grid_test_sun_directions,
    #         grid_test_ae,
    #     ) = cached_generate_grid_sun_array(
    #         cfg.TEST.SUN_DIRECTIONS,
    #         device,
    #         case="grid",
    #     )
    #     grid_test_targets = cached_generate_grid_dataset(
    #         H_validation,
    #         ENV_validation,
    #         grid_test_sun_directions,
    #         None,
    #         "grid",
    #     )
    #     # # th.random.set_rng_state(state)
    #     H_naive_grid = build_target_heliostat(cfg, device)
    #     H_naive_grid._normals = H_naive_grid._normals_ideal
    #     naive_targets = cached_generate_naive_grid_dataset(
    #         H_naive_grid,
    #         ENV_validation,
    #         grid_test_sun_directions,
    #         None,
    #         "naive",
    #     )
    # if cfg.TEST.PLOT.SPHERIC:
    #     (
    #         spheric_test_sun_directions,
    #         spheric_test_ae,
    #     ) = cached_generate_spheric_sun_array(
    #         cfg.TEST.SUN_DIRECTIONS,
    #         device,
    #         train_vec=sun_directions,
    #         case="spheric",
    #     )
    #     spheric_test_targets = cached_generate_spheric_dataset(
    #         H_validation,
    #         ENV_validation,
    #         spheric_test_sun_directions,
    #         None,
    #         "spheric",
    #     )

    #     H_naive_spheric = build_target_heliostat(cfg, device)
    #     H_naive_spheric._normals = H_naive_spheric._normals_ideal
    #     naive_spheric_test_targets = cached_generate_naive_spheric_dataset(
    #         H_naive_spheric,
    #         ENV_validation,
    #         spheric_test_sun_directions,
    #         None,
    #         "naive_spheric",
    #     )
    # if cfg.TEST.PLOT.SEASON:
    #     (
    #         season_test_sun_directions,
    #         season_test_extras,
    #     ) = cached_generate_season_sun_array(
    #         cfg.TEST.SUN_DIRECTIONS,
    #         device,
    #         case="season",
    #     )
    #     season_test_sun_directions = season_test_sun_directions.to(device) #TODO bring to GPU in data.py
    #     season_test_targets = cached_generate_season_dataset(
    #         H_validation,
    #         ENV_validation,
    #         season_test_sun_directions,
    #         None,
    #         None,
    #         "season_"
    #     )
    #     H_naive_season = build_target_heliostat(cfg, device)
    #     H_naive_season._normals = H_naive_season._normals_ideal
    #     naive_season_test_targets = cached_generate_naive_season_dataset(
    #         H_naive_season,
    #         ENV_validation,
    #         season_test_sun_directions,
    #         None,
    #         None,
    #         "naive_season_"
    #     )

    # plotter.test_surfaces(H_target)
    # exit()
    # Start Diff Raytracing
    # =====================
    print("Initialize Diff Raytracing")
    print(f"Use {cfg.NURBS.ROWS}x{cfg.NURBS.COLS} NURBS")
    print("=============================")
    H = build_heliostat(cfg, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

#pretraining 
    # pretrain_epochs = 2000
    # steps_per_epoch = int(th.ceil(th.tensor(pretrain_epochs / len(targets))))
    # opt, sched = build_optimizer_scheduler(
    #     cfg, pretrain_epochs * steps_per_epoch, H.get_params(), device)
    # loss_func, test_loss_func = build_loss_funcs(cfg.TRAIN.LOSS)
    # epoch_shift_width = len(str(pretrain_epochs))
    # best_result = th.tensor(float('inf'))
    # prefix = 'pretrain'
    plotter.plot_surfaces_3D_mm(H, 999999, logdir_surfaces, writer = None)
    # for epoch in range(pretrain_epochs):
    #     train_objects = TrainObjects(
    #         opt,
    #         sched,
    #         H,
    #         ENV,
    #         R,
    #         naive_targets,
    #         sun_directions,
    #         loss_func,
    #         epoch,
    #         writer,
    #         prefix
    #     )

    #     if writer:
    #         writer.add_scalar(f"{prefix}/lr", opt.param_groups[0]["lr"], epoch)
            
    #     loss, pred_bitmap, num_missed = train_batch(train_objects)
    #     print(
    #         f'Pretraining [{epoch:>{epoch_shift_width}}/{pretrain_epochs}] '
    #         f'loss: {loss.detach().cpu().numpy()}, '
    #         f'lr: {opt.param_groups[0]["lr"]:.2e}, '
    #         f'missed: {num_missed.detach().cpu().item()}, '
    #     )
    #     if epoch % 15 ==0:
    #         # test_loss, _ = test_batch(
    #         #     H,
    #         #     ENV,
    #         #     R,
    #         #     test_targets,
    #         #     test_sun_directions,
    #         #     test_loss_func,
    #         #     epoch,
    #         #     writer,
    #         #     "pretest"
    #         # )
    # plotter.plot_surfaces_mrad(
    #     H_naive_target,
    #     H,
    #     epoch,
    #     logdir_pretrain_surfaces,
    #     None
    # )

    epochs: int = cfg.TRAIN.EPOCHS
    steps_per_epoch = int(th.ceil(th.tensor(epochs / len(targets))))

    opt, sched = build_optimizer_scheduler(
        cfg, epochs * steps_per_epoch, H.get_params(), device)
    loss_func, test_loss_func = build_loss_funcs(cfg.TRAIN.LOSS)

    epoch_shift_width = len(str(epochs))

    best_result = th.tensor(float('inf'))

    # Generate naive Losses before training
    # spheric_naive_test_loss, _ = test_batch(
    #                 H,
    #                 ENV,
    #                 R,
    #                 spheric_test_targets,
    #                 spheric_test_sun_directions,
    #                 test_loss_func,
    #                 0,
    #                 reduction=False
    #             )
    
    # season_naive_test_loss, _ = test_batch(
    #                 H,
    #                 ENV,
    #                 R,
    #                 season_test_targets,
    #                 season_test_sun_directions,
    #                 test_loss_func,
    #                 0,
    #                 reduction=False
    #             )
    

    prefix = "train"
    for epoch in range(epochs):
        train_objects = TrainObjects(
            opt,
            sched,
            H,
            ENV,
            R,
            targets,
            sun_directions,
            loss_func,
            epoch,
            prefix,
            writer,
        )
        # if epoch == 0:
        #     plotter.plot_surfaces_3D_mm(H, 100000, logdir_surfaces, writer = None)
        loss, pred_bitmap, num_missed = train_batch(train_objects)
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'lr: {opt.param_groups[0]["lr"]:.2e}, '
            f'missed: {num_missed.detach().cpu().item()}, '
        )
        if writer:
            writer.add_scalar(f"{prefix}/lr", opt.param_groups[0]["lr"], epoch)

            if epoch % cfg.TEST.INTERVAL == 0:
                test_loss, _ = test_batch(
                    H,
                    ENV,
                    R,
                    test_targets,
                    test_sun_directions,
                    test_loss_func,
                    epoch,
                    "test",
                    writer,
                )
                plotter.plot_surfaces_mrad(
                    H_target,
                    H,
                    epoch,
                    logdir_surfaces,
                    writer,
                )
                # if epoch != 0:
                # season_test_loss, season_test_bitmaps = test_batch(
                #     H,
                #     ENV,
                #     R,
                #     season_test_targets,
                #     season_test_sun_directions,
                #     test_loss_func,
                #     epoch,
                #     'test_season',
                #     reduction=False,
                # )

                # plotter.season_plot(
                #     season_test_extras,
                #     naive_season_test_targets,
                #     season_test_bitmaps,
                #     season_test_targets,
                #     season_test_loss,
                #     season_naive_test_loss,
                #     logdir_enhanced_test,
                #     epoch,
                # )
                plotter.plot_surfaces_3D_mm(
                    H, epoch, logdir_surfaces, writer=None)
            #     grid_test_loss, grid_test_bitmaps = test_batch(
            #         H,
            #         ENV,
            #         R,
            #         grid_test_targets,
            #         grid_test_sun_directions,
            #         test_loss_func,
            #         epoch,
            #         'test_grid',
            #     )

            #     plotter.target_image_comparision_pred_orig_naive(
            #         grid_test_ae,
            #         grid_test_targets,
            #         grid_test_bitmaps,
            #         naive_targets,
            #         sun_directions,
            #         epoch,
            #         logdir_enhanced_test,
            #     )

            #     spheric_train_loss, _ = test_batch(
            #         H,
            #         ENV,
            #         R,
            #         targets,
            #         sun_directions,
            #         test_loss_func,
            #         epoch,
            #         'test_train',
            #         reduction=False,
            #     )
            #     spheric_test_loss, spheric_test_bitmaps = test_batch(
            #         H,
            #         ENV,
            #         R,
            #         spheric_test_targets,
            #         spheric_test_sun_directions,
            #         test_loss_func,
            #         epoch,
            #         'test_spheric',
            #         reduction=False,
            #     )
            #     plotter.spherical_loss_plot(
            #         sun_directions,
            #         spheric_test_ae,
            #         spheric_train_loss,
            #         spheric_test_loss,
            #         spheric_naive_test_loss,
            #         cfg.TEST.SUN_DIRECTIONS.SPHERIC.NUM_SAMPLES,
            #         epoch,
            #         logdir_enhanced_test,
            #     )

                print(
                    f'[{epoch:>{epoch_shift_width}}/{epochs}] '
                    f'test loss: {test_loss.item()}'
                )

            # Plotting stuff
            # if test_loss.detach().cpu() < best_result and cfg.SAVE_RESULTS:
            #     # plotter.target_image_comparision_pred_orig_naive(
            #     #     test_ae,
            #     #     test_targets,
            #     #     test_bitmaps,
            #     #     naive_targets,
            #     #     sun_directions,
            #     #     epoch,
            #     #     logdir_enhanced_test,
            #     # )

            #     plotter.plot_surfaces_mm(
            #         H_target,
            #         H,
            #         epoch,
            #         logdir_surfaces,
            #         writer
            #     )
                # plotter.plot_surfaces_3D_mm(
                #     H, epoch, logdir_surfaces, writer=None)
                # plotter.plot_surfaces_3D_mrad(
                #     H_target, H, epoch, logdir_surfaces, writer=None)

        # Save Section

        if test_loss.detach().cpu() < best_result:
            if cfg.SAVE_RESULTS:
                # Remember best checkpoint data (to store on disk later).
                save_data = H.to_dict()
                save_data['xi'] = R.xi
                save_data['yi'] = R.yi
                opt_save_data = {'opt': copy.deepcopy(opt.state_dict())}
                # Store remembered data and optimizer state on disk.
                model_name = type(H).__name__
                assert logdir_files is not None
                th.save(
                    save_data,
                    os.path.join(logdir_files, f'{model_name}.pt'),
                )
                th.save(
                    opt_save_data,
                    os.path.join(logdir_files, f'{model_name}_opt.pt'),
                )
            best_result = test_loss.detach().cpu()

    # Diff Raytracing >


if __name__ == '__main__':
    path_to_yaml = os.path.join("TestingConfigs", "DeepPretrain.yaml")
    main(path_to_yaml)
    # main()
