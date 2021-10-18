import os

import matplotlib.pyplot as plt
import torch as th

from defaults import get_cfg_defaults, load_config_file
from environment import Environment
from heliostat_models import Heliostat
from nurbs_heliostat import NURBSHeliostat
# import plotter
from render import Renderer
import utils


def main():
    # < Initialization
    # Load Defaults
    # =============
    load_default = True
    # not used if `load_default is True`
    config_file = os.path.join("configs", "LoadDeflecData.yaml")
    # not used if `load_default is True`
    experiment_name = 'first_test_new'

    cfg_default = get_cfg_defaults()
    if load_default:
        cfg = cfg_default
    else:
        cfg = load_config_file(cfg_default, config_file, experiment_name)

    cfg.freeze()

    # Set up Logging
    # =============
    logdir = os.path.join(cfg.LOGDIR, cfg.ID)
    cfg.merge_from_list(["LOGDIR", logdir])
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # writer = SummaryWriter(logdir)
    # Write out config parameters.

    # Set system params
    # =================
    th.manual_seed(cfg.SEED)
    device = th.device(
        'cuda'
        if cfg.USE_GPU and th.cuda.is_available()
        else 'cpu'
    )

    # Set up Environment
    # =================
    # Create Heliostat Object and Load Model defined in config file
    H = Heliostat(cfg.H, device)
    ENV = Environment(cfg.AC, device)
    target_save_data = (
        H.position_on_field,
        th.tensor(
            H.cfg.IDEAL.NORMAL_VECS,
            dtype=th.get_default_dtype(),
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

        th.tensor(
            ENV.cfg.SUN.ORIGIN,
            dtype=th.get_default_dtype(),
            device=device,
        ),
        ENV.sun.num_rays,
        ENV.sun.mean,
        ENV.sun.cov,
    )
    H.align(ENV.sun_origin, ENV.receiver_center)
    R = Renderer(H, ENV)
    utils.save_target(
        *(
            target_save_data
            + (
                R.xi,
                R.yi,

                # We need the heliostat to be aligned here.
                H.get_ray_directions(),
                H.discrete_points,
                'target.pt',
            )
        )
    )
    del target_save_data

    # Render Step
    # ===========
    target_bitmap = R.render()
    targets = target_bitmap.detach().clone().unsqueeze(0)

    # Plot and Save Stuff
    # ===================
    im = plt.imshow(target_bitmap.detach().cpu().numpy(), cmap='jet')
    plt.savefig(os.path.join("images", "original.jpeg"))
    # plotter.plot_bitmap(target_bitmap)  # Target Bitmap Plot

    # Delete Setup
    # ============
    del H
    del ENV
    del R
    # Initialization >

    # TODO Bis hierhin fertig refactored
    # < Diff Raytracing
    # TODO Load other Constants than in Setup
    if cfg.CP_PATH is not None and cfg.CP_PATH != '':
        cp = th.load(cfg.CP_PATH, map_location=device)
        if cfg.USE_NURBS:
            H = NURBSHeliostat.from_dict(cp, device)
        else:
            H = Heliostat.from_dict(cp, device)
    else:
        if cfg.USE_NURBS:
            H = NURBSHeliostat(cfg.H, cfg.NURBS, device)
        else:
            # Create Heliostat Object and Load Model defined in config file
            H = Heliostat(cfg.H, device)
    ENV = Environment(cfg.AC, device)
    R = Renderer(H, ENV)

    if cfg.USE_NURBS:
        opt = th.optim.Adam(H.setup_params(), lr=3e-6, weight_decay=0.1)
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-12,
            patience=10,
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
        opt = th.optim.Adam(H.setup_params(), lr=3e-4, weight_decay=0.1)
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-12,
            patience=10,
            verbose=True,
        )

    # loss = th.nn.functional.mse_loss()
    # def loss_func(pred, target, compute_intersections, rayPoints):
    #     loss = th.nn.functional.mse_loss(pred, target, 0.1)
    #     # if cfg.USE_CURL:
    #     #     curls = th.stack([
    #     #         curl(compute_intersections, rayPoints_)
    #     #         for rayPoints_ in rayPoints
    #     #     ])
    #     #     loss += th.sum(th.abs(curls))
    #     return loss

    epochs = cfg.TRAIN_PARAMS.EPOCHS
    epoch_shift_width = len(str(epochs))
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = 0
        H.align(ENV.sun_origin, ENV.receiver_center, verbose=False)
        # print(ray_directions)
        for target in targets:
            pred_bitmap = R.render()
            if epoch % 10 == 0:
                im.set_data(pred_bitmap.detach().cpu().numpy())
                im.autoscale()
                plt.savefig(os.path.join("images", f"{epoch}.png"))
            loss += th.nn.functional.mse_loss(pred_bitmap, target)
            # loss += loss_func(
            #     pred_bitmap,
            #     target,
            #     lambda rayPoints: compute_receiver_intersections(
            #         planeNormal,
            #         aimpoint,
            #         ray_directions,
            #         rayPoints,
            #         xi,
            #         yi,
            #     ),
            #     rayPoints,
            # )
        # if epoch %  10== 0:#
        #     im.set_data(pred.detach().cpu().numpy())
        #     im.autoscale()
        #     plt.savefig(os.path.join("images", f"{epoch}.png"))

        loss /= len(targets)
        loss.backward()
        # if not use_splines:
        #     if (
        #             ray_directions.grad is None
        #             or (ray_directions.grad == 0).all()
        #     ):
        #         print('no more optimization possible; ending...')
        #         break

        opt.step()
        if isinstance(sched, th.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(loss)
        else:
            sched.step()
        with th.no_grad():
            num_missed = R.indices.numel() - R.indices.count_nonzero()
            ray_diff = utils.calc_ray_diffs(
                R.ray_directions,
                H.get_ray_directions().detach(),
            )
            print(
                f'[{epoch:>{epoch_shift_width}}/{epochs}] '
                f'loss: {loss.detach().cpu().numpy()}, '
                f'missed: {num_missed.detach().cpu().item()}, '
                f'ray differences: {ray_diff.detach().cpu().item()}'
            )
            H.align_reverse()
    # Diff Raytracing >

    # Save trained model and optimizer state
    save_data = H.to_dict()
    save_data['xi'] = R.xi
    save_data['yi'] = R.yi
    model_name = type(H).__name__
    th.save(save_data, f'{model_name}.pt')
    th.save({'opt': opt.state_dict()}, f'{model_name}_opt.pt')


if __name__ == '__main__':
    main()
