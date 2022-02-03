import collections
import copy
from datetime import datetime
import os

import matplotlib.pyplot as plt
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

import data
from defaults import get_cfg_defaults, load_config_file
from environment import Environment
from heliostat_models import Heliostat
from multi_nurbs_heliostat import MultiNURBSHeliostat
from nurbs_heliostat import NURBSHeliostat
import plotter
from render import Renderer
import utils


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
        'writer',
    ],
    # 'writer' is None by default
    defaults=[None],
)


def check_consistency(cfg):
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


def load_heliostat(cfg, device):
    cp_path = os.path.expanduser(cfg.CP_PATH)
    cp = th.load(cp_path, map_location=device)
    if cfg.USE_NURBS:
        H = NURBSHeliostat.from_dict(
            cp,
            device,
            nurbs_config=cfg.NURBS,
            config=cfg.H,
        )
    else:
        H = Heliostat.from_dict(cp, device)
    return H


def load_optimizer_state(opt, cp_path, device):
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


def _build_multi_nurbs_target(cfg, device):
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


def _multi_nurbs_to_standard(cfg, mnh):
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


def build_target_heliostat(cfg, device):
    if cfg.H.SHAPE.lower() == 'nurbs':
        mnh = _build_multi_nurbs_target(cfg, device)
        H = _multi_nurbs_to_standard(cfg, mnh)
    else:
        H = Heliostat(cfg.H, device)
    return H


def build_heliostat(cfg, device):
    if cfg.CP_PATH and os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
        H = load_heliostat(cfg, device)
    else:
        if cfg.USE_NURBS:
            if (
                    cfg.NURBS.FACETS.POSITIONS is not None
                    and len(cfg.NURBS.FACETS.POSITIONS) > 1
            ):
                nurbs_heliostat_cls = MultiNURBSHeliostat
                kwargs = {'receiver_center': cfg.AC.RECEIVER.CENTER}
            else:
                nurbs_heliostat_cls = NURBSHeliostat
                kwargs = {}

            H = nurbs_heliostat_cls(cfg.H, cfg.NURBS, device, **kwargs)
        else:
            H = Heliostat(cfg.H, device)
    return H


def _build_optimizer(cfg_optimizer, params):
    cfg = cfg_optimizer
    name = cfg.NAME.lower()

    if name == "adam":
        opt = th.optim.Adam(
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


def _build_scheduler(cfg_scheduler, opt, total_steps):
    name = cfg_scheduler.NAME.lower()
    if name == "reduceonplateu":
        cfg = cfg_scheduler.ROP
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
    elif name == "onecycle":
        cfg = cfg_scheduler.ONE_CYCLE
        sched = th.optim.lr_scheduler.OneCycleLR(
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


def _get_opt_cp_path(cp_path):
    return os.path.expanduser(cp_path[:-3] + '_opt.pt')


def build_optimizer_scheduler(cfg, total_steps, params, device):
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params)
    # Load optimizer state.
    if cfg.LOAD_OPTIMIZER_STATE:
        opt_cp_path = _get_opt_cp_path(cfg.CP_PATH)
        load_optimizer_state(opt, opt_cp_path, device)

    sched = _build_scheduler(cfg.TRAIN.SCHEDULER, opt, total_steps)
    return opt, sched


def build_loss_funcs(cfg_loss):
    cfg = cfg_loss
    name = cfg.NAME.lower()
    if name == "mse":
        test_loss_func = th.nn.MSELoss()
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


def calc_batch_loss(train_objects, return_extras=True):
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
        writer,
    ) = train_objects
    # Initialize Parameters
    # =====================
    loss = 0
    if return_extras:
        num_missed = 0.0
        ray_diff = 0

    # Batch Loop
    # ==========
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        H_aligned = H.align(sun_direction, ENV.receiver_center)
        pred_bitmap, (ray_directions, indices, _, _) = R.render(
            H_aligned, return_extras=True)
        loss += loss_func(pred_bitmap, target, opt) / len(targets)

        with th.no_grad():
            # Plot target images to TensorBoard
            if writer:
                if epoch % 50 == 0:
                    writer.add_image(
                        f"target_{i}/prediction",
                        utils.colorize(pred_bitmap),
                        epoch,
                    )

            if return_extras:
                # Compare metrics
                num_missed += (
                    (indices.numel() - indices.count_nonzero())
                    / len(targets)
                )
                ray_diff += utils.calc_ray_diffs(
                    ray_directions,
                    H_aligned.get_ray_directions().detach(),
                ) / len(targets)

    if return_extras:
        return loss, (pred_bitmap, num_missed, ray_diff)
    return loss


def calc_batch_grads(train_objects, return_extras=True):
    train_objects.opt.zero_grad(set_to_none=True)

    loss, (pred_bitmap, num_missed, ray_diff) = calc_batch_loss(
        train_objects, return_extras=True)

    loss.backward()
    if return_extras:
        return loss, (pred_bitmap, num_missed, ray_diff)
    return loss


def train_batch(train_objects):
    opt = train_objects.opt
    if isinstance(opt, th.optim.LBFGS):
        with th.no_grad():
            _, (pred_bitmap, num_missed, ray_diff) = calc_batch_loss(
                train_objects)
        loss = opt.step(
            lambda: calc_batch_grads(train_objects, return_extras=False),
        )
    else:
        loss, (pred_bitmap, num_missed, ray_diff) = calc_batch_grads(
            train_objects)
        opt.step()

    # Plot loss to Tensorboard
    with th.no_grad():
        writer = train_objects.writer
        if writer:
            writer.add_scalar("train/loss", loss.item(), train_objects.epoch)

    # Update training parameters
    # ==========================
    sched = train_objects.sched
    if isinstance(sched, th.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(loss)
    else:
        sched.step()

    # if opt.param_groups[0]["lr"] < last_lr:
    train_objects.H.step(verbose=True)
    #     last_lr = opt.param_groups[0]["lr"]

    return loss, pred_bitmap, num_missed, ray_diff


@th.no_grad()
def test_batch(
        heliostat,
        env,
        renderer,
        targets,
        sun_directions,
        loss_func,
        epoch,
        writer=None,
):
    loss = 0
    for (i, (target, sun_direction)) in enumerate(zip(
            targets,
            sun_directions,
    )):
        heliostat_aligned = heliostat.align(
            sun_direction, env.receiver_center)
        pred_bitmap = renderer.render(heliostat_aligned)
        loss += loss_func(pred_bitmap, target) / len(targets)

        if writer:
            writer.add_image(
                            f"test_target_{i}/prediction",
                            utils.colorize(pred_bitmap),
                            epoch,
                        )

    if writer:
        writer.add_scalar("test/loss", loss.item(), epoch)

    return loss


def main(config_file_name=None):
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

    # Set up Logging
    # ==============
    if cfg.SAVE_RESULTS:
        now = datetime.now()
        time_str = now.strftime("%y%m%d_%H%M")
        root_logdir = os.path.join(cfg.LOGDIR, cfg.ID)
        logdir = os.path.join(
            root_logdir,
            cfg.EXPERIMENT_NAME + f"_{time_str}",
        )
        logdir_files = os.path.join(logdir, "Logfiles")
        logdir_images = os.path.join(logdir, "Images")
        logdir_diffs = os.path.join(logdir_images, "Diffs")
        logdir_surfaces = os.path.join(logdir_images, "Surfaces")
        cfg.merge_from_list(["LOGDIR", logdir])
        os.makedirs(root_logdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(logdir_files, exist_ok=True)
        os.makedirs(logdir_images, exist_ok=True)
        os.makedirs(logdir_diffs, exist_ok=True)

        with open(os.path.join(logdir, "config.yaml"), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)

        writer = SummaryWriter(logdir)
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

    # Create Dataset
    # ==============
    # Create Heliostat Object and Load Model defined in config file
    print("Create dataset using:")
    print(f"Sun direction(s): {cfg.AC.SUN.DIRECTION}")
    print(f"Aimpoint: {cfg.AC.RECEIVER.CENTER}")
    print(
        f"Receiver Resolution: {cfg.AC.RECEIVER.RESOLUTION_X}×"
        f"{cfg.AC.RECEIVER.RESOLUTION_Y}"
    )
    print("=============================")
    H_target = build_target_heliostat(cfg, device)

    # plotter.plot_normal_vectors(H_target.discrete_points, H_target.normals)

    ENV = Environment(cfg.AC, device)
    targets, sun_directions = data.generate_dataset(
        cfg.AC.SUN.DIRECTION,
        H_target,
        ENV,
        logdir_files,
        writer,
    )
    # plt.imshow(targets[0].cpu().detach().squeeze())

    test_targets, test_sun_directions = data.generate_test_dataset(
        cfg.TEST,
        H_target,
        ENV,
        None,
        writer,
    )
    # plotter.test_surfaces(H_target)
    # exit()
    # Start Diff Raytracing
    # =====================
    print("Initialize Diff Raytracing")
    print(f"Use {cfg.NURBS.ROWS}x{cfg.NURBS.COLS} NURBS")
    print("=============================")
    H = build_heliostat(cfg, device)
    # plotter.test_surfaces(H)
    # plotter.plot_normal_vectors(
    # H._discrete_points, H._normals)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

    # plotter.plot_normal_vectors(H.discrete_points, H.normals)
    # plotter.test_surfaces(H)
    # plt.imshow(targets.cpu().detach().squeeze())

    epochs = cfg.TRAIN.EPOCHS
    steps_per_epoch = int(th.ceil(th.tensor(epochs / len(targets))))

    opt, sched = build_optimizer_scheduler(
        cfg, epochs * steps_per_epoch, H.get_params(), device)
    loss_func, test_loss_func = build_loss_funcs(cfg.TRAIN.LOSS)

    epoch_shift_width = len(str(epochs))

    best_result = th.tensor(float('inf'))
    state_dict = {
        "epoch": 0,
        "current_loss": float('inf'),
        "best_loss": float('inf'),
        "last_lr": opt.param_groups[0]["lr"],
        "current_lr": opt.param_groups[0]["lr"],
    }
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
            writer,
        )
        loss, pred_bitmap, num_missed, ray_diff = train_batch(train_objects)

        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'lr: {opt.param_groups[0]["lr"]:.2e}, '
            f'missed: {num_missed.detach().cpu().item()}, '
            f'ray differences: {ray_diff.detach().cpu().item()}'
        )
        if writer:
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)

        if epoch % cfg.TEST.INTERVAL == 0:
            test_loss = test_batch(
                H,
                ENV,
                R,
                test_targets,
                test_sun_directions,
                test_loss_func,
                epoch,
                writer,
            )
            print(
                f'[{epoch:>{epoch_shift_width}}/{epochs}] '
                f'test loss: {test_loss.item()}'
            )

        if epoch % 50 == 0 and cfg.SAVE_RESULTS:

            plotter.plot_surfaces_mrad(
                H_target,
                H,
                epoch,
                logdir_surfaces,
                writer
            )
            plotter.plot_surfaces_mm(
                H_target,
                H,
                epoch,
                logdir_surfaces,
                writer
            )

        # Save Section
        if cfg.SAVE_RESULTS:
            if loss.detach().cpu() < best_result:
                # Remember best checkpoint data (to store on disk later).
                best_result = loss.detach().cpu()
                save_data = H.to_dict()
                save_data['xi'] = R.xi
                save_data['yi'] = R.yi
                opt_save_data = {'opt': copy.deepcopy(opt.state_dict())}
                found_new_best_result = True
            # if epoch % 100 == 0 and found_new_best_result:
                # Store remembered data and optimizer state on disk.
                model_name = type(H).__name__
                th.save(
                    save_data,
                    os.path.join(logdir_files, f'{model_name}.pt'),
                )
                th.save(
                    opt_save_data,
                    os.path.join(logdir_files, f'{model_name}_opt.pt'),
                )
                found_new_best_result = False

    # Diff Raytracing >


if __name__ == '__main__':
    main()
