import atexit
import copy
from datetime import datetime
import functools
import os
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

import data
from defaults import get_cfg_defaults, load_config_file
import disk_cache
from environment import Environment
import facets
import hausdorff_distance
from heliostat_models import AbstractHeliostat, Heliostat
from multi_nurbs_heliostat import MultiNURBSHeliostat, NURBSFacets
from nurbs_heliostat import AbstractNURBSHeliostat, NURBSHeliostat
import plotter
from render import Renderer
import training
import utils


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
            and not os.path.isfile(training.get_opt_cp_path(cfg.CP_PATH))
    ):
        warnings_found = True
        print(
            "WARNING: Optimizer checkpoint not found; "
            "continuing without loading..."
        )

    nurbs_focus_point = cfg.NURBS.FACETS.CANTING.FOCUS_POINT
    heliostat_cfg = Heliostat.select_heliostat_builder(cfg.H)[1]
    heliostat_focus_point = heliostat_cfg.FACETS.CANTING.FOCUS_POINT
    if (
            nurbs_focus_point != 'inherit'
            and nurbs_focus_point != heliostat_focus_point
    ):
        warnings_found = True
        print(
            "WARNING: Focus points of target and trained heliostat "
            "do not match!"
        )

    if not warnings_found:
        print("No warnings found. Good Luck!")
        print("=============================")


def set_up_dataset_caching(
        device: th.device,
        writer: Optional[SummaryWriter],
) -> Tuple[
    Tuple[Callable, Callable, Callable, Callable, Callable],
    Tuple[
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
    ],
]:
    def make_cached_generate_sun_array(
            prefix: str = '',
    ) -> Callable[
        [CfgNode, th.device, Optional[torch.Tensor], Optional[str]],
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]],
    ]:
        return disk_cache.disk_cache(
            data.generate_sun_array,
            device,
            'cached',
            prefix,
            ignore_argnums=[1],
        )

    def make_cached_generate_dataset(
            prefix: str,
            tb_log: bool = True,
    ) -> Callable[
        [
            AbstractHeliostat,
            Environment,
            torch.Tensor,
            Optional[str],
            str,
            Optional[SummaryWriter],
        ],
        torch.Tensor,
    ]:
        if tb_log:
            log_dataset: Optional[disk_cache.OnLoadFn] = functools.partial(
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


def load_heliostat(
        cfg: CfgNode,
        sun_directions: torch.Tensor,
        device: th.device,
) -> AbstractHeliostat:
    cp_path = os.path.expanduser(cfg.CP_PATH)
    cp = th.load(cp_path, map_location=device)
    if cfg.USE_NURBS:
        if 'facets' in cp:
            nurbs_heliostat_cls: Type[AbstractNURBSHeliostat] = \
                MultiNURBSHeliostat
        else:
            nurbs_heliostat_cls = NURBSHeliostat
        H: AbstractHeliostat = nurbs_heliostat_cls.from_dict(
            cp,
            device,
            nurbs_config=cfg.NURBS,
            config=cfg.H,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
        )
    else:
        H = Heliostat.from_dict(
            cp,
            device,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
        )
    return H


def _build_multi_nurbs_target(
        cfg: CfgNode,
        sun_directions: torch.Tensor,
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
        sun_directions=sun_directions,
        setup_params=False,
    )

    assert isinstance(mnh.facets, NURBSFacets)
    for facet in mnh.facets:
        assert isinstance(facet, NURBSHeliostat)
        facet.set_ctrl_points(
            facet.ctrl_points
            + th.rand_like(facet.ctrl_points)
            * mnh_cfg.H.NURBS.MAX_ABS_NOISE
        )

    return mnh


def _multi_nurbs_to_standard(
        cfg: CfgNode,
        sun_directions: torch.Tensor,
        mnh: MultiNURBSHeliostat,
) -> Heliostat:
    H = Heliostat(
        cfg.H,
        mnh.device,
        receiver_center=cfg.AC.RECEIVER.CENTER,
        sun_directions=sun_directions,
        setup_params=False,
    )
    discrete_points, normals = mnh.discrete_points_and_normals()

    H.facets = facets.Facets(
        H,
        mnh.facets.positions,
        mnh.facets.spans_n,
        mnh.facets.spans_e,
        mnh.facets.raw_discrete_points,
        mnh.facets.raw_discrete_points_ideal,
        mnh.facets.raw_normals,
        mnh.facets.raw_normals_ideal,
        mnh.facets.cant_rots,
    )
    H.params = mnh.nurbs_cfg
    H.height = mnh.height
    H.width = mnh.width
    H.rows = mnh.rows
    H.cols = mnh.cols

    return H


def build_target_heliostat(
        cfg: CfgNode,
        sun_directions: torch.Tensor,
        device: th.device,
) -> Heliostat:
    if cfg.H.SHAPE.lower() == 'nurbs':
        mnh = _build_multi_nurbs_target(cfg, sun_directions, device)
        H = _multi_nurbs_to_standard(cfg, sun_directions, mnh)
    else:
        H = Heliostat(
            cfg.H,
            device,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
            setup_params=False,
        )
    return H


def build_heliostat(
        cfg: CfgNode,
        sun_directions: torch.Tensor,
        device: th.device,
) -> AbstractHeliostat:
    if cfg.CP_PATH and os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
        H = load_heliostat(cfg, sun_directions, device)
    else:
        if cfg.USE_NURBS:
            H = MultiNURBSHeliostat(
                cfg.H,
                cfg.NURBS,
                device,
                receiver_center=cfg.AC.RECEIVER.CENTER,
                sun_directions=sun_directions,
            )
        else:
            H = Heliostat(
                cfg.H,
                device,
                receiver_center=cfg.AC.RECEIVER.CENTER,
                sun_directions=sun_directions,
            )
    return H


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

    utils.fix_pytorch3d()

    # Set up Logging
    # ==============
    if cfg.SAVE_RESULTS:
        now = datetime.now()
        time_str = now.strftime("%y%m%d_%H%M")

        logdir: Optional[str] = cfg.LOGDIR
        assert logdir is not None
        logdir = utils.normalize_path(logdir)

        root_logdir = os.path.join(logdir, cfg.ID)
        logdir = os.path.join(
            root_logdir,
            cfg.EXPERIMENT_NAME + f"_{time_str}",
        )
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
        logdir: Optional[str]
    else:
        writer = None
        logdir = None
        logdir_files = None
        logdir_images = None

    if isinstance(writer, SummaryWriter):
        atexit.register(lambda: cast(SummaryWriter, writer).close())

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
            cached_generate_season_sun_array,
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
    cached_build_target_heliostat = cast(
        Callable[[CfgNode, torch.Tensor, th.device], Heliostat],
        disk_cache.disk_cache(
            build_target_heliostat,
            device,
            'cached',
            ignore_argnums=[2],
        ),
    )

    # Create Dataset
    # ==============
    # Create Heliostat Object and Load Model defined in config file
    print("Create dataset using:")
    print(f"Receiver Center: {cfg.AC.RECEIVER.CENTER}")
    print(
        f"Receiver Resolution: {cfg.AC.RECEIVER.RESOLUTION_X}×"
        f"{cfg.AC.RECEIVER.RESOLUTION_Y}"
    )
    print("=============================")
    sun_directions, ae = cached_generate_sun_array(
        cfg.TRAIN.SUN_DIRECTIONS, device)
    H_target = cached_build_target_heliostat(cfg, sun_directions, device)
    ENV = Environment(cfg.AC, device)

    if cfg.TRAIN.USE_IMAGES:
        assert cfg.TRAIN.SUN_DIRECTIONS.CASE.lower() == 'vecs', (
            'must have known sun directions for training with images '
            '(set `CASE` to "vec" and check the given `VECS.DIRECTIONS`)'
        )
        assert len(cfg.TRAIN.IMAGES.PATHS) == len(sun_directions), \
            'number of sun directions does not match number of images'
        targets = data.load_images(
            cfg.TRAIN.IMAGES.PATHS,
            cfg.AC.RECEIVER.RESOLUTION_X,
            cfg.AC.RECEIVER.RESOLUTION_Y,
            device,
            'train',
            writer,
        )
    else:
        targets = cached_generate_target_dataset(
            H_target,
            ENV,
            sun_directions,
            logdir_files,
            "train",
            writer,
        )
    target_z_alignments = utils.get_z_alignments(H_target, sun_directions)
    if cfg.TRAIN.LOSS.HAUSDORFF.FACTOR != 0:
        data.log_contoured(
            'train',
            writer,
            targets,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
        )
        target_sets: Optional[
            List[torch.Tensor],
        ] = hausdorff_distance.images_to_sets(
            targets,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
        )
    else:
        target_sets = None

    H_naive_target = cached_build_target_heliostat(cfg, sun_directions, device)
    H_naive_target._normals = H_naive_target.get_raw_normals_ideal()
    naive_targets = cached_generate_pretrain_dataset(
        H_naive_target,
        ENV,
        sun_directions,
        logdir_files,
        "pretrain",
        writer,
    )
    naive_target_z_alignments = utils.get_z_alignments(
        H_naive_target, sun_directions)
    if cfg.TRAIN.LOSS.HAUSDORFF.FACTOR != 0:
        naive_target_sets: Optional[
            List[torch.Tensor],
        ] = hausdorff_distance.images_to_sets(
            naive_targets,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
            cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
        )
    else:
        naive_target_sets = None

    test_sun_directions, test_ae = cached_generate_test_sun_array(
        cfg.TEST.SUN_DIRECTIONS, device)

    if cfg.TEST.USE_IMAGES:
        assert cfg.TEST.SUN_DIRECTIONS.CASE.lower() == 'vecs', (
            'must have known sun directions for testing with images '
            '(set `CASE` to "vec" and check the given `VECS.DIRECTIONS`)'
        )
        assert len(cfg.TEST.IMAGES.PATHS) == len(test_sun_directions), \
            'number of sun directions does not match number of images'
        test_targets = data.load_images(
            cfg.TEST.IMAGES.PATHS,
            cfg.AC.RECEIVER.RESOLUTION_X,
            cfg.AC.RECEIVER.RESOLUTION_Y,
            device,
            'test',
            writer,
        )
    else:
        test_targets = cached_generate_test_dataset(
            H_target,
            ENV,
            test_sun_directions,
            None,
            "test",
            writer,
        )
    # plotter.plot_surfaces_mm(H_target, H_target, 1, logdir)
    test_target_sets = hausdorff_distance.images_to_sets(
        test_targets,
        cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
        cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
    )

    # Better Testing
    # ==============
    # state = th.random.get_rng_state()

    if cfg.TEST.PLOT.GRID or cfg.TEST.PLOT.SPHERIC or cfg.TEST.PLOT.SEASON:
        H_validation = cached_build_target_heliostat(
            cfg, sun_directions, device)
        ENV_validation = Environment(cfg.AC, device)
    if cfg.TEST.PLOT.GRID:
        (
            grid_test_sun_directions,
            grid_test_ae,
        ) = cached_generate_grid_sun_array(
            cfg.TEST.SUN_DIRECTIONS,
            device,
            case="grid",
        )
        grid_test_targets = cached_generate_grid_dataset(
            H_validation,
            ENV_validation,
            grid_test_sun_directions,
            None,
            "grid",
        )
        # # th.random.set_rng_state(state)
        H_naive_grid = cached_build_target_heliostat(
            cfg, sun_directions, device)
        H_naive_grid._normals = H_naive_grid.get_raw_normals_ideal()
        naive_grid_targets = cached_generate_naive_grid_dataset(
            H_naive_grid,
            ENV_validation,
            grid_test_sun_directions,
            None,
            "naive",
        )
    if cfg.TEST.PLOT.SPHERIC:
        (
            spheric_test_sun_directions,
            spheric_test_ae,
        ) = cached_generate_spheric_sun_array(
            cfg.TEST.SUN_DIRECTIONS,
            device,
            train_vec=sun_directions,
            case="spheric",
        )
        spheric_test_targets = cached_generate_spheric_dataset(
            H_validation,
            ENV_validation,
            spheric_test_sun_directions,
            None,
            "spheric",
        )

        H_naive_spheric = cached_build_target_heliostat(
            cfg, sun_directions, device)
        H_naive_spheric._normals = H_naive_spheric.get_raw_normals_ideal()
        naive_spheric_test_targets = cached_generate_naive_spheric_dataset(
            H_naive_spheric,
            ENV_validation,
            spheric_test_sun_directions,
            None,
            "naive_spheric",
        )
    if cfg.TEST.PLOT.SEASON:
        (
            season_test_sun_directions,
            season_test_extras,
        ) = cached_generate_season_sun_array(
            cfg.TEST.SUN_DIRECTIONS,
            device,
            case="season",
        )
        # TODO bring to GPU in data.py
        season_test_sun_directions = season_test_sun_directions.to(device)
        season_test_targets = cached_generate_season_dataset(
            H_validation,
            ENV_validation,
            season_test_sun_directions,
            None,
            "season",
        )
        H_naive_season = cached_build_target_heliostat(
            cfg, sun_directions, device)
        H_naive_season._normals = H_naive_season.get_raw_normals_ideal()
        naive_season_test_targets = cached_generate_naive_season_dataset(
            H_naive_season,
            ENV_validation,
            season_test_sun_directions,
            None,
            "naive_season",
        )

    # plotter.test_surfaces(H_target)
    # exit()
    # Start Diff Raytracing
    # =====================
    print("Initialize Diff Raytracing")
    print(f"Use {cfg.NURBS.ROWS}x{cfg.NURBS.COLS} NURBS")
    print("=============================")
    H = build_heliostat(cfg, sun_directions, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

    # Pretraining
    # ===========

    pretrain_epochs: int = cfg.TRAIN.PRETRAIN_EPOCHS
    steps_per_epoch = int(th.ceil(th.tensor(pretrain_epochs / len(targets))))
    opt, sched = training.build_optimizer_scheduler(
        cfg,
        pretrain_epochs * steps_per_epoch,
        H.get_params(),
        device,
    )
    loss_func, test_loss_func = training.build_loss_funcs(
        cfg.TRAIN.LOSS, H.get_to_optimize())
    epoch_shift_width = len(str(pretrain_epochs))
    best_result = th.tensor(float('inf'))
    prefix = 'pretrain'
    for epoch in range(pretrain_epochs):
        plotter.plot_surfaces_3D_mm(
            H,
            epoch,
            logdir_pretrain_surfaces,
            writer=None,
        )
        train_objects = training.TrainObjects(
            opt,
            sched,
            H,
            ENV,
            R,
            naive_targets,
            naive_target_z_alignments,
            naive_target_sets,
            sun_directions,
            loss_func,
            cfg,
            epoch,
            prefix,
            writer,
        )

        if writer:
            writer.add_scalar(
                f"{prefix}/lr", opt.param_groups[0]["lr"], epoch)

        loss, raw_loss, pred_bitmap, num_missed = training.train_batch(
            train_objects)
        print(
            f'Pretraining [{epoch:>{epoch_shift_width}}/{pretrain_epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'raw loss: {raw_loss.detach().cpu().numpy()}, '
            f'lr: {opt.param_groups[0]["lr"]:.2e}, '
            f'missed: {num_missed.detach().cpu().item()}, '
        )
        if epoch % 15 == 0:
            plotter.plot_surfaces_3D_mm(
                H,
                epoch,
                logdir_pretrain_surfaces,
                writer=None,
            )
            # plotter.plot_surfaces_mrad(
            #     H_target,
            #     H,
            #     777777,
            #     logdir_surfaces,
            #     writer,
            # )
            # test_loss, _ = test_batch(
            #     H,
            #     ENV,
            #     R,
            #     test_targets,
            #     test_sun_directions,
            #     test_loss_func,
            #     epoch,
            #     "pretest",
            #     writer,
            # )
    # plotter.plot_surfaces_mrad(
    #     H_naive_target,
    #     H,
    #     epoch,
    #     logdir_pretrain_surfaces,
    #     None
    # )

    epochs: int = cfg.TRAIN.EPOCHS
    steps_per_epoch = int(th.ceil(th.tensor(epochs / len(targets))))

    opt, sched = training.build_optimizer_scheduler(
        cfg,
        epochs * steps_per_epoch,
        H.get_params(),
        device,
    )
    loss_func, test_loss_func = training.build_loss_funcs(
        cfg.TRAIN.LOSS, H.get_to_optimize())

    epoch_shift_width = len(str(epochs))

    best_result = th.tensor(float('inf'))

    # Generate naive Losses before training
    # spheric_naive_test_loss, _ = test_batch(
    #     H,
    #     ENV,
    #     R,
    #     spheric_test_targets,
    #     spheric_test_sun_directions,
    #     test_loss_func,
    #     0,
    #     'naive_spheric',
    #     reduction=False,
    # )

    # season_naive_test_loss, _ = test_batch(
    #     H,
    #     ENV,
    #     R,
    #     season_test_targets,
    #     season_test_sun_directions,
    #     test_loss_func,
    #     0,
    #     'naive_season',
    #     reduction=False,
    # )
    plotter.plot_surfaces_mrad(
        H_target,
        H,
        9797979797,
        logdir_surfaces,
        writer=None,
    )
    prefix = "train"
    for epoch in range(epochs):
        train_objects = training.TrainObjects(
            opt,
            sched,
            H,
            ENV,
            R,
            targets,
            target_z_alignments,
            target_sets,
            sun_directions,
            loss_func,
            cfg,
            epoch,
            prefix,
            writer,
        )
        # if epoch == 0:
        #     plotter.plot_surfaces_3D_mm(
        #         H, 100000, logdir_surfaces, writer=None)
        loss, raw_loss, pred_bitmap, num_missed = training.train_batch(
            train_objects)
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'raw loss: {raw_loss.detach().cpu().numpy()}, '
            f'lr: {opt.param_groups[0]["lr"]:.2e}, '
            f'missed: {num_missed.detach().cpu().item()}, '
        )
        if writer:
            writer.add_scalar(f"{prefix}/lr", opt.param_groups[0]["lr"], epoch)

            if epoch % cfg.TEST.INTERVAL == 0:
                test_loss, hausdorff_dist, _ = training.test_batch(
                    H,
                    ENV,
                    R,
                    test_targets,
                    test_target_sets,
                    test_sun_directions,
                    test_loss_func,
                    cfg,
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
                #     epoch,
                #     logdir_enhanced_test,
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
            #         naive_grid_targets,
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
                    f'test loss: {test_loss.item()}, '
                    f'Hausdorff distance: {hausdorff_dist.item()}'
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
    path_to_yaml = os.path.join("TestingConfigs", "600m_Function.yaml")
    main(path_to_yaml)
    # main()
