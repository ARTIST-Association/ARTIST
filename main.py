import atexit
import copy
from datetime import datetime
import os
from typing import Callable, cast, List, Optional

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

import bpro_loader
from build_heliostat_model import build_heliostat, build_target_heliostat
import data
import dataset_cache
from defaults import get_cfg_defaults, load_config_file
import disk_cache
from environment import Environment
import hausdorff_distance
from heliostat_models import Heliostat
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


def main(
        config_file_name: Optional[str] = None,
        sweep: Optional[bool] = False,
) -> None:
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
        logdir: Optional[str]
        if sweep:
            assert config_file_name is not None
            logdir = os.path.split(config_file_name)[0]
        else:
            now = datetime.now()
            time_str = now.strftime("%y%m%d_%H%M%S")

            logdir = cfg.LOGDIR
            assert logdir is not None
            logdir = utils.normalize_path(logdir)

            root_logdir = os.path.join(logdir, cfg.ID)
            os.makedirs(root_logdir, exist_ok=True)

            logdir = os.path.join(
                root_logdir,
                cfg.EXPERIMENT_NAME + f"_{time_str}",
            )
            os.makedirs(logdir, exist_ok=True)
            with open(os.path.join(logdir, "config.yaml"), "w") as f:
                f.write(cfg.dump())

        assert logdir is not None
        logdir_files: Optional[str] = os.path.join(logdir, "Logfiles")
        assert logdir_files is not None
        logdir_images: Optional[str] = os.path.join(logdir, "Images")
        assert logdir_images is not None
        logdir_enhanced_test = os.path.join(logdir_images, "EnhancedTest")
        logdir_surfaces = os.path.join(logdir_images, "Surfaces")

        os.makedirs(logdir, exist_ok=True)
        os.makedirs(logdir_files, exist_ok=True)
        os.makedirs(logdir_images, exist_ok=True)
        os.makedirs(logdir_enhanced_test, exist_ok=True)

        writer: Optional[SummaryWriter] = SummaryWriter(logdir)
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
        ),
        (
            cached_generate_target_dataset,
            cached_generate_pretrain_dataset,
            cached_generate_test_dataset,
        ),
    ) = dataset_cache.set_up_dataset_caching(device, writer)
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
    print(f"Heliostat position on field: {H_target.position_on_field}")
    ENV = Environment(cfg.AC, device)

    if cfg.TRAIN.USE_IMAGES:
        assert cfg.TRAIN.SUN_DIRECTIONS.CASE.lower() == 'vecs', (
            'must have known sun directions for training with images '
            '(set `CASE` to "vec" and check the given `VECS.DIRECTIONS`)'
        )
        assert len(cfg.TRAIN.IMAGES.PATHS) == len(sun_directions), \
            'number of sun directions does not match number of images'
        targets = data.load_images(
            list(map(utils.normalize_path, cfg.TRAIN.IMAGES.PATHS)),
            cfg.AC.RECEIVER.PLANE_X,
            cfg.AC.RECEIVER.PLANE_Y,
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

    #     from matplotlib import pyplot as plt #TODO Remove for release
    #     plt.imshow(targets.squeeze(), cmap="gray")
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #                 hspace = 0, wspace = 0)
    #     plt.margins(0,0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     # plt.show()
    #     plt.savefig(
    #         str(sun_directions.squeeze()) + ".png",
    #         bbox_inches='tight',
    #         pad_inches=0,
    #     )
    #     exit()

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
    # H_naive_target = cached_build_target_heliostat(
    #     cfg, sun_directions, device)
    # H_naive_target._normals = H_naive_target.get_raw_normals_ideal()
    # naive_targets = cached_generate_pretrain_dataset(
    #     H_naive_target,
    #     ENV,
    #     sun_directions,
    #     logdir_files,
    #     "pretrain",
    #     writer,
    # )
    # naive_target_z_alignments = utils.get_z_alignments(
    #     H_naive_target, sun_directions)

    # if cfg.TRAIN.LOSS.HAUSDORFF.FACTOR != 0:
    #     naive_target_sets: Optional[
    #         List[torch.Tensor],
    #     ] = hausdorff_distance.images_to_sets(
    #         naive_targets,
    #         cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
    #         cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
    #     )
    # else:
    #     naive_target_sets = None

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
            list(map(utils.normalize_path, cfg.TEST.IMAGES.PATHS)),
            cfg.AC.RECEIVER.PLANE_X,
            cfg.AC.RECEIVER.PLANE_Y,
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
    test_target_sets = hausdorff_distance.images_to_sets(
        test_targets,
        cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
        cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
    )

    # Start Diff Raytracing
    # =====================
    print("Initialize Diff Raytracing")
    print(f"Use {cfg.NURBS.ROWS}x{cfg.NURBS.COLS} NURBS")
    print("=============================")

    # Pretraining
    # ===========
    pretrain_epochs: int = cfg.TRAIN.PRETRAIN_EPOCHS
    steps_per_epoch = 1
    epoch_shift_width = len(str(pretrain_epochs))
    prefix = 'pretrain'

    # for (facet, normals) in zip(
    #         H.facets._facets,
    #         H_target._normals,
    # ):
    #     facet._normals = normals
    # alignment_params = []
    # found_something = False
    # for i, sun in enumerate(sun_directions):
    #     H = build_heliostat(cfg, sun.unsqueeze(0), device)
    #     ENV = Environment(cfg.AC, device)
    #     R = Renderer(H, ENV)
    #     H.set_to_optimize(["rotation_x","rotation_y","rotation_z"])
    #     loss_func, test_loss_func = training.build_loss_funcs(
    #         cfg.TRAIN.LOSS, H.get_to_optimize())
    #     best_pretrain_loss = th.tensor(float('inf'))
    #     opt, sched = training.build_optimizer_scheduler(
    #         cfg,
    #         pretrain_epochs * steps_per_epoch,
    #         H.get_params(),
    #         device,
    #     )
    #     # H._normals = H_target._normals

    #     for epoch in range(pretrain_epochs):
    #         train_objects = training.TrainObjects(
    #             opt,
    #             sched,
    #             H,
    #             ENV,
    #             R,
    #             targets[i].unsqueeze(0),
    #             target_z_alignments[i].unsqueeze(0),
    #             target_sets,
    #             sun.unsqueeze(0),
    #             loss_func,
    #             cfg,
    #             epoch,
    #             prefix,
    #             writer,
    #             None,
    #             i,
    #         )
    #         loss, raw_loss, pred_bitmap, num_missed = training.train_batch(
    #             train_objects)
    #         if raw_loss < best_pretrain_loss:
    #             found_something = True
    #             print("found new best alignment")
    #             best_angles = H.disturbance_angles
    #             best_pretrain_loss = raw_loss
    #             utils.to_tensorboard(
    #                 writer,
    #                 prefix,
    #                 epoch,
    #                 image=pred_bitmap,
    #                 plot_interval=cfg.TRAIN.IMG_INTERVAL,
    #                 index=i,
    #             )

    #         print(
    #             f'Pretraining '
    #             f'[{epoch:>{epoch_shift_width}}/{pretrain_epochs}] '
    #             f'loss: {loss.detach().cpu().numpy()}, '
    #             f'raw loss: {raw_loss.detach().cpu().numpy()}, '
    #             # f'lr: {opt.param_groups[0]["lr"]:.2e}, '
    #             f'missed: {num_missed.detach().cpu().item()}, '
    #         )

    #         utils.to_tensorboard(writer, prefix, epoch,
    #                              lr=opt.param_groups[0]["lr"],
    #                              loss=loss,
    #                              raw_loss=raw_loss,
    #                              index=i
    #                              )
    #     if found_something:
    #         print(best_angles)
    #         alignment_params.append(best_angles)
    # print(alignment_params)

    epochs: int = cfg.TRAIN.EPOCHS
    steps_per_epoch = 1
    H = build_heliostat(cfg, sun_directions, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)
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

    prefix = "train"
    # real_data.yaml
    # Converged pre-aligment from pretraining. It is hardcoded here for
    # faster tests with pre-aligment. TODO Remove later.
    # prealignment = [
    #     [th.tensor(0.0038), th.tensor(0.0010), th.tensor(-0.7040)],
    #     [th.tensor(-0.0243), th.tensor(-0.0014), th.tensor(-0.6096)],
    #     [th.tensor(0.0127), th.tensor(0.0108), th.tensor(0.0631)],
    #     [th.tensor(-0.0140), th.tensor(0.0085), th.tensor(-0.7635)]
    #             ]
    # test_prealignment = [
    #     [th.tensor(-0.0112), th.tensor(-0.0050), th.tensor(-0.8203)]
    #     ]

    # real_data_2.yaml
    # Converged pre-aligment from pretraining. It is hardcoded here for
    # faster tests with pre-aligment. TODO Remove later.
    # prealignment = [
    #     [th.tensor(-0.0140), th.tensor(0.0085), th.tensor(-0.7635)],
    #     [th.tensor(0.0127), th.tensor(0.0108), th.tensor(0.0631)],
    #             ]
    # test_prealignment = [
    #     [th.tensor(0.0038), th.tensor(0.0010), th.tensor(-0.7040)],
    #     [th.tensor(-0.0243), th.tensor(-0.0014), th.tensor(-0.6096)],
    #     ]
    prealignment = None
    test_prealignment = None

    # Better Testing
    # =============
    plot = plotter.Plotter(
        cfg,
        R,
        sun_directions,
        test_loss_func,
        device,
        train_prealignment=prealignment,
        test_prealignment=test_prealignment,
    )

    # Clear caches
    bpro_loader.load_bpro.cache_clear()
    bpro_loader.load_csv.cache_clear()

    for epoch in range(epochs):
        test_objects = training.TestObjects(
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
            H_target,
            logdir_surfaces,
            True,
            test_prealignment,
        )
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
            test_objects,
            None,
            prealignment,
        )

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
                    test_objects)
                utils.to_tensorboard(
                    writer,
                    'test',
                    epoch,
                    loss=test_loss,
                    cfg=cfg,
                )
                print(
                    f'[{epoch:>{epoch_shift_width}}/{epochs}] '
                    f'test loss: {test_loss.item()}, '
                    f'Hausdorff distance: {hausdorff_dist.item()}'
                )

            # Plotting stuff
            # if test_loss.detach().cpu() < best_result and cfg.SAVE_RESULTS:
            if epoch % cfg.TEST.INTERVAL == 0:
                plotter.plot_surfaces_mrad(
                    H_target,
                    H,
                    epoch,
                    logdir_surfaces,
                    writer,
                )
                # for i, image in enumerate(pred_bitmap):

            utils.to_tensorboard(
                writer,
                prefix,
                epoch,
                lr=opt.param_groups[0]["lr"],
                loss=loss,
                raw_loss=raw_loss,
                cfg=cfg,
            )

        # Save Section
        # Advanced Plotting
        plot.create_plots(H, epoch, logdir_enhanced_test, H_target=H_target)

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
    if sweep:
        # new_name = "_" + os.path.split(logdir)[-1]
        # new_dir = os.path.join(os.path.split(logdir)[0], new_name)
        # os.rename(logdir, new_dir)
        print("Sweep Instance finished")


if __name__ == '__main__':
    path_to_yaml = os.path.join(
        "TestingConfigs", "BJ25.yaml")
    main(path_to_yaml)
    # main()
