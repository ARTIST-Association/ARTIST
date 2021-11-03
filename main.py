import copy
from datetime import datetime
import os

import torch as th
from torch.utils.tensorboard import SummaryWriter

import data
from defaults import get_cfg_defaults, load_config_file
from environment import Environment
from heliostat_models import Heliostat
from nurbs_heliostat import NURBSHeliostat
import plotter
from render import Renderer
import utils


def check_consistency(cfg):
    print("Loaded Switches:")
    print(f"Heliostat shape: {cfg.H.SHAPE}")
    print(f"Solar distribustion: {cfg.AC.SUN.DISTRIBUTION}")
    print(f"Scheduler: {cfg.TRAIN.SCHEDULER.NAME}")
    print(f"Optimizer: {cfg.TRAIN.OPTIMIZER.NAME}")
    print(f"Loss: {cfg.TRAIN.LOSS.NAME}")

    warnings_found = False
    if cfg.TRAIN.LOSS.USE_L1_WEIGHT_DECAY:
        if not cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY == 0:
            warnings_found = True
            print("WARNING: Do you realy want to use L2 and L1 Weight Decay?")
    if cfg.TRAIN.SCHEDULER.NAME.lower() == "cyclic":
        if not cfg.TRAIN.SCHEDULER.CYCLIC.BASE_LR == cfg.TRAIN.OPTIMIZER.LR:
            warnings_found = True
            print(
                "WARNING: Cyclic base lr and optimizer lr should be the same")
    if not warnings_found:
        print("No warnings found. Good Luck!")
        print("=============================")


def load_heliostat(cfg, device):
    cp = th.load(os.path.expanduser(cfg.CP_PATH), map_location=device)
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


def build_heliostat(cfg, device):
    if cfg.CP_PATH:
        H = load_heliostat(cfg, device)
    else:
        if cfg.USE_NURBS:
            H = NURBSHeliostat(cfg.H, cfg.NURBS, device)
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
        opt = th.optim.Adam(
            params,
            lr=cfg.LR,
            betas=(cfg.BETAS[0], cfg.BETAS[1]),
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        raise ValueError(
            "Optimizer name not found, change name or implement new optimizer")

    return opt


def _build_scheduler(cfg_scheduler, opt):
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
        cfg = cfg_scheduler.Cyclic
        sched = th.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=cfg.BASE_LR,
            max_lr=cfg.MAX_LR,
            step_size_up=cfg.STEP_SIZE_UP,
            cycle_momentum=cfg.CYCLE_MOMENTUM,
            mode=cfg.MODE,
        )
    elif name == "onecycle":
        cfg = cfg.scheduler.ONE_CYCLE
        sched = th.optim.lr_scheduler.OneCycleLR(
            opt,
            total_steps=cfg.TOTAL_STEPS,
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


def build_optimizer_scheduler(cfg, params, device):
    opt = _build_optimizer(cfg.TRAIN.OPTIMIZER, params)
    # Load optimizer state.
    if cfg.LOAD_OPTIMIZER_STATE:
        opt_cp_path = cfg.CP_PATH[:-3] + '_opt.pt'
        load_optimizer_state(opt, opt_cp_path, device)

    sched = _build_scheduler(cfg.TRAIN.SCHEDULER, opt)
    return opt, sched


def loss_func(cfg_loss, pred_bitmap, target, opt):
    cfg = cfg_loss
    name = cfg.NAME.lower()
    if name == "mse":
        loss = th.nn.functional.mse_loss(pred_bitmap, target)
    elif name == "l1":
        loss = th.nn.functional.l1_loss(pred_bitmap, target)
    else:
        raise ValueError(
            "Loss function name not found, change name or implement new loss")

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


def train_batch(
        cfg_train,
        opt,
        sched,
        H,
        ENV,
        R,
        targets,
        sun_origins,
        epoch,
        writer=None,
):
    # Initialize Parameters
    # =====================
    # Reset cache so we don't use cached but reset gradients.
    H.reset_cache()
    opt.zero_grad(set_to_none=True)
    loss = 0
    num_missed = 0.0
    ray_diff = 0

    # Batch Loop
    # ==========
    for (i, (target, sun_origin)) in enumerate(zip(
            targets,
            sun_origins,
    )):
        ENV.sun_origin = sun_origin
        H.align(ENV.sun_origin, ENV.receiver_center, verbose=False)
        pred_bitmap = R.render()
        # loss += th.nn.functional.l1_loss(pred_bitmap, target)/len(targets)
        loss += (
            loss_func(cfg_train.LOSS, pred_bitmap, target, opt)
            / len(targets)
        )

        # Plot target images to TensorBoard
        if writer:
            if epoch % 50 == 0:
                writer.add_image(
                    f"target_{i}/prediction",
                    utils.colorize(pred_bitmap),
                    epoch,
                )

        # Compare metrics
        with th.no_grad():
            H.align_reverse()
            num_missed += (
                (R.indices.numel() - R.indices.count_nonzero())
                / len(targets)
            )
            ray_diff += utils.calc_ray_diffs(
                R.ray_directions,
                H.get_ray_directions().detach(),
            ) / len(targets)

    # Plot loss to Tensorboard
    if writer:
        writer.add_scalar("train/loss", loss.item(), epoch)

    # Update training parameters
    # ==========================
    loss.backward()
    opt.step()
    if isinstance(sched, th.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(loss)
    else:
        sched.step()
    H.step(verbose=True)

    return loss, pred_bitmap, num_missed, ray_diff


def main():
    # Load Defaults
    # =============
    config_file_name = None
    cfg_default = get_cfg_defaults()
    if config_file_name:
        config_file = os.path.join("configs", config_file_name)
        cfg = load_config_file(cfg_default, config_file)
    else:
        cfg = cfg_default
    cfg.freeze()

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
        os.makedirs(logdir_surfaces, exist_ok=True)
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
    print(f"Sun position(s): {cfg.AC.SUN.ORIGIN}")
    print(f"Aimpoint: {cfg.AC.RECEIVER.CENTER}")
    print(
        f"Receiver Resolution: {cfg.AC.RECEIVER.RESOLUTION_X}×"
        f"{cfg.AC.RECEIVER.RESOLUTION_Y}"
    )
    print("=============================")
    H_target = Heliostat(cfg.H, device)
    ENV = Environment(cfg.AC, device)
    targets, sun_origins = data.generate_dataset(
        cfg,
        H_target,
        ENV,
        logdir_files,
        writer,
    )

    # Start Diff Raytracing
    # =====================
    print("Initialize Diff Raytracing")
    print(f"Use {cfg.H.NURBS.ROWS}x{cfg.H.NURBS.COLS} NURBS")
    print("=============================")
    H = build_heliostat(cfg, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

    opt, sched = build_optimizer_scheduler(cfg, H.setup_params(), device)

    epochs = cfg.TRAIN.EPOCHS
    epoch_shift_width = len(str(epochs))

    best_result = th.tensor(float('inf'))
    for epoch in range(epochs):
        loss, pred_bitmap, num_missed, ray_diff = train_batch(
            cfg.TRAIN,
            opt,
            sched,
            H,
            ENV,
            R,
            targets,
            sun_origins,
            epoch,
            writer,
        )

        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'lr: {opt.param_groups[0]["lr"]:.2e}, '
            f'missed: {num_missed.detach().cpu().item()}, '
            f'ray differences: {ray_diff.detach().cpu().item()}'
        )
        if writer:
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)

        if epoch % 50 == 0 and cfg.SAVE_RESULTS:
            plotter.plot_surfaces(
                H_target._discrete_points_orig,
                th.tile(
                    th.tensor([0, 0, 1], device=device),
                    (H_target._discrete_points_orig.shape[0], 1),
                ),
                H_target._normals_orig,
                H._normals_orig,
                epoch,
                logdir_surfaces,
                writer
            )
            # plotter.plot_diffs(
            #     H_target._discrete_points_orig,
            #     th.tile(
            #         th.tensor([0, 0, 1], device=device),
            #         (H_target._discrete_points_orig.shape[0], 1),
            #     ),
            #     H_target._normals_orig,
            #     H._normals_orig,
            #     epoch,
            #     logdir_diffs,
            # )

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
            if epoch % 100 == 0 and found_new_best_result:
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
