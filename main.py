import os


import torch as th

from datetime import datetime

from defaults import get_cfg_defaults, load_config_file
from environment import Environment
from heliostat_models import Heliostat
from nurbs_heliostat import NURBSHeliostat
from torch.utils.tensorboard import SummaryWriter
import plotter
import matplotlib.pyplot as plt
from render import Renderer
import utils

def main():
    # < Initialization
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
    now = datetime.now()
    time_str =now.strftime("%y%m%d_%H%M")
    logdir = os.path.join(cfg.LOGDIR, cfg.ID+"_"+time_str)
    logdir_files = os.path.join(logdir, "Logfiles")
    logdir_images = os.path.join(logdir, "Images")
    cfg.merge_from_list(["LOGDIR", logdir])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir_files, exist_ok=True)
    os.makedirs(logdir_images,exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    writer = SummaryWriter(logdir) 
    
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
    targets = None
    sun_origins = cfg.AC.SUN.ORIGIN
    if not isinstance(sun_origins[0], list):
        sun_origins = [sun_origins]
    sun_origins = th.tensor(
        sun_origins, dtype=th.get_default_dtype(), device=device)
    sun_origins_normed = sun_origins / sun_origins.norm(dim=1).unsqueeze(-1)

    H_target = Heliostat(cfg.H, device)
    ENV = Environment(cfg.AC, device)
    for (i, (sun_origin, sun_origin_normed)) in enumerate(zip(
            sun_origins,
            sun_origins_normed
    )):
        ENV.sun_origin = sun_origin_normed
        target_save_data = (
            H_target.position_on_field,
            th.tensor(
                H_target.cfg.IDEAL.NORMAL_VECS,
                dtype=th.get_default_dtype(),
                device=device,
            ),
            H_target.discrete_points,
            H_target.normals,
            None,  # TODO

            ENV.receiver_center,
            ENV.receiver_plane_x,
            ENV.receiver_plane_y,
            ENV.receiver_plane_normal,
            None,  # TODO

            sun_origin,
            ENV.sun.num_rays,
            ENV.sun.mean,
            ENV.sun.cov,
        )
        H_target.align(ENV.sun_origin, ENV.receiver_center, verbose=(i == 0))
        R = Renderer(H_target, ENV)
        utils.save_target(
            *(
                target_save_data
                + (
                    R.xi,
                    R.yi,

                    # We need the heliostat to be aligned here.
                    H_target.get_ray_directions(),
                    H_target.discrete_points,
                    f'target_{i}.pt',
                )
            )
        )
        del target_save_data

        # Render Step
        # ===========
        target_bitmap = R.render()
        if targets is None:
            targets = th.empty(
                (len(sun_origins),) + target_bitmap.shape,
                dtype=th.get_default_dtype(),
                device=device,
            )
        targets[i] = target_bitmap
        H_target.align_reverse()

        # Plot and Save Stuff
        # ===================
        # print(H._normals_orig.shape)
        writer.add_image("originals", utils.colorize(target_bitmap))
        # im = plt.imshow(target_bitmap.detach().cpu(),cmap = "jet")
    del sun_origins

    # Initialization >
    
    # TODO Bis hierhin fertig refactored
    # < Diff Raytracing
    # TODO Load other Constants than in Setup
    load_cp = cfg.CP_PATH is not None and cfg.CP_PATH != ''
    if load_cp:
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
        """
        In order to avoid
        getting stuck in local minima, we perform the optimization in a
        multi-scale fashion, starting from 64 ×64 and linearly increasing to
        """
        """
        reset stark ausgelagerte NURBS
        """
        """
        Mehrere Sonnenstände
        """
        
        """
        We exclude the light source
        in the loss function by setting the weights of pixels with radiance
        larger than 5 to 0
        """
        
        opt = th.optim.Adamax(H.setup_params(), lr=2e-4, weight_decay=0.2)#
        # sched = th.optim.lr_scheduler.CyclicLR(opt,base_lr=1e-8, max_lr=2e-4, step_size_up=250, cycle_momentum=False,mode="triangular2" )
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-7,
            patience=200,
            cooldown=400,
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
        opt = th.optim.Adam(H.setup_params(), lr=3e-6, weight_decay=0.1)
        sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            min_lr=1e-12,
            patience=10,
            
            verbose=True,
        )

    # Load optimizer state.
    if load_cp:
        opt_cp_path = cfg.CP_PATH[:-3] + '_opt.pt'
        if not os.path.isfile(opt_cp_path):
            print(
                f'Warning: cannot find optimizer under {opt_cp_path}; '
                f'please rename your optimizer checkpoint accordingly. '
                f'Continuing with newly created optimizer...'
            )
        cp = th.load(opt_cp_path, map_location=device)
        opt.load_state_dict(cp['opt'])
        del cp

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
    
    best_result = th.tensor(float('inf'))
    last_save= 0
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = 0
        num_missed = 0.0
        ray_diff = 0
        # print(ray_directions)
        for (i, (target, sun_origin)) in enumerate(zip(
                targets,
                sun_origins_normed,
        )):
            ENV.sun_origin = sun_origin
            H.align(ENV.sun_origin, ENV.receiver_center, verbose=False)
            pred_bitmap = R.render()

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

            with th.no_grad():
                H.align_reverse()
                num_missed += R.indices.numel() - R.indices.count_nonzero()
                ray_diff += utils.calc_ray_diffs(
                    R.ray_directions,
                    H.get_ray_directions().detach(),
                )
        # if epoch %  10== 0:#
        #     im.set_data(pred.detach().cpu().numpy())
        #     im.autoscale()
        #     plt.savefig(os.path.join("images", f"{epoch}.png"))

        loss /= len(targets)
        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 50 == 0:
            writer.add_image("prediction", utils.colorize(pred_bitmap), epoch)
            
            
    
            
    
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
            num_missed /= len(targets)
            ray_diff /= len(targets)
            print(
                f'[{epoch:>{epoch_shift_width}}/{epochs}] '
                f'loss: {loss.detach().cpu().numpy()}, '
                f'missed: {num_missed.detach().cpu().item()}, '
                f'ray differences: {ray_diff.detach().cpu().item()}'
            )
            H.align_reverse()

        if epoch %50 ==0:
            plotter.plot_surface_diff(H_target._discrete_points_orig, th.tile(th.tensor([0,0,1], device=device),(1024,1)), H_target._normals_orig,  H._normals_orig, epoch, logdir_images)
        # if epoch % 1 == 0:
        #     num_missed = indices.numel() - indices.count_nonzero()
        #     ray_diff = calc_ray_diffs(
        #         ray_directions.detach(),
        #         target_ray_directions,
        #     )
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
    
            f'lr: {opt.param_groups[0]["lr"]:.2e}'
                        # f'lr: {sched.get_lr()}'
            # f'missed: {num_missed.detach().cpu().item()}, '
            # f'ray differences: {ray_diff.detach().cpu().item()}'
        )
        if loss.detach().cpu() < best_result:
            best_result = loss.detach().cpu()
            save_data = H.to_dict()
            save_data['xi'] = R.xi
            save_data['yi'] = R.yi
        if epoch % 100 == 0:
            model_name = type(H).__name__
            th.save(save_data, os.path.join(logdir_files, f'{model_name}.pt'))
            th.save(
                {'opt': opt.state_dict()},
                os.path.join(logdir_files, f'{model_name}_opt.pt'),
            )

    # Diff Raytracing >



if __name__ == '__main__':
    main()
