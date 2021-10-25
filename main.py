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


def load_heliostat(cfg, device):
    cp = th.load(cfg.CP_PATH, map_location=device)
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


def build_optimizer(cfg, params, device):
    if cfg.USE_NURBS:
        # print(params)
        opt = th.optim.Adamax(params, lr=1e-4, weight_decay=0.2)
        # sched = th.optim.lr_scheduler.CyclicLR(
        #     opt,
        #     base_lr=1e-8,
        #     max_lr=8.3e-5,
        #     step_size_up=100,
        #     cycle_momentum=False,
        #     mode="triangular2",
        # )
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-8,
            patience=100,
            cooldown=200,
            verbose=True,
        )
        # max_lr = 6e-5
        # start_lr = 1e-10
        # final_lr = 1e-8
        # sched = th.optim.lr_scheduler.OneCycleLR(
        #     opt,
        #     total_steps=cfg.TRAIN_PARAMS.EPOCHS,
        #     max_lr=max_lr,
        #     pct_start=0.1,
        #     div_factor=max_lr / start_lr,
        #     final_div_factor=max_lr / final_lr,
        #     # three_phase=True,
        # )
    else:
        opt = th.optim.Adam(params, lr=3e-6, weight_decay=0.1)
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-12,
            patience=10,

            verbose=True,
        )

    # Load optimizer state.
    if cfg.CP_PATH:
        opt_cp_path = cfg.CP_PATH[:-3] + '_opt.pt'
        load_optimizer_state(opt, opt_cp_path, device)
    return opt, sched


def train_batch(
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

        loss += th.nn.functional.l1_loss(pred_bitmap, target) / len(targets)

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
        logdir = os.path.join(cfg.LOGDIR, cfg.ID+"_"+time_str)
        logdir_files = os.path.join(logdir, "Logfiles")
        logdir_images = os.path.join(logdir, "Images")
        logdir_diffs = os.path.join(logdir_images, "Diffs")
        logdir_surfaces = os.path.join(logdir_images, "Surfaces")
        cfg.merge_from_list(["LOGDIR", logdir])
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
    H = build_heliostat(cfg, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

    opt, sched = build_optimizer(cfg, H.setup_params(), device)

    epochs = cfg.TRAIN_PARAMS.EPOCHS
    epoch_shift_width = len(str(epochs))

    best_result = th.tensor(float('inf'))
    for epoch in range(epochs):
        loss, pred_bitmap, num_missed, ray_diff = train_batch(
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
        # if not use_splines:
        #     if (
        #             ray_directions.grad is None
        #             or (ray_directions.grad == 0).all()
        #     ):
        #         print('no more optimization possible; ending...')
        #         break

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
                save_data = copy.deepcopy(H.to_dict())
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
