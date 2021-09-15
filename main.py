import sys
import torch as th
import os

##My Files
sys.path.insert(0, 'configs')

from defaults import get_cfg_defaults, load_config_file
from heliostat_models import Heliostat
from environment import Environment
from render import Renderer
from plotter import plot_surface_diff, plot_normal_vectors, plot_raytracer, plot_heliostat, plot_bitmap
import matplotlib.pyplot as plt
from matplotlib import animation
import nurbs


def main():
    # < Initilization
    ### Load Defaults
    load_default = True
    config_file     = "configs\\LoadDeflecData.yaml" # not used if load_default is True
    experiment_name= 'first_test_new'  # not used if load_default is True

    cfg_default             = get_cfg_defaults()
    if load_default:
        cfg = cfg_default
    else:
        cfg = load_config_file(cfg_default, config_file, experiment_name)

    cfg.freeze()

    ### Setup Loggin
    logdir = os.path.join(cfg.LOGDIR, cfg.ID)
    cfg.merge_from_list(["LOGDIR", logdir])
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # writer = SummaryWriter(logdir)
    # Write out config parameters.

    ###Set system params
    th.manual_seed(cfg.SEED)
    device = th.device('cuda' if cfg.USE_GPU and th.cuda.is_available() else 'cpu')


    ###Setup Envoirment
    H = Heliostat(cfg.H, device) #Creat Heliostat Object and Load Model defined in config file
    ENV = Environment(cfg.AC, device)
    H.align(ENV.sun_origin, ENV.receiver_center)
    R = Renderer(H, ENV)

    ###Render Step
    target_bitmap = R.render()
    targets = target_bitmap.detach().clone().unsqueeze(0)

    ###Plot and Save Stuff
    im = plt.imshow(target_bitmap.detach().cpu().numpy(), cmap='jet')
    im.set_data(target_bitmap.detach().cpu().numpy())
    im.autoscale()
    plt.savefig(os.path.join("images", "original.jpeg"))
    # plot_bitmap(target_bitmap)                                                 # Target Bitmap Plot

    ###Delete Setup
    del H
    del ENV
    del R
    # Initilization >



    ###### Bis hierhin fertig refactored
    # < Diff Raytracing
    #TODO Load other Constants than in Setup
    H = Heliostat(cfg.H, device) #Creat Heliostat Object and Load Model defined in config file
    ENV = Environment(cfg.AC, device)
    H.align(ENV.sun_origin, ENV.receiver_center)
    R = Renderer(H, ENV)

    R.ray_directions.requires_grad_(True)
    opt = th.optim.Adam([R.ray_directions], lr=3e-4, weight_decay=0.1)
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
        opt.zero_grad()
        loss = 0
        # print(ray_directions)
        for target in targets:
            # if use_splines:
            #     ctrl_points = th.hstack((ctrl_points_xy, ctrl_points_z))
            #     hel_origin, surface_normals = (
            #         nurbs.calc_normals_and_surface_slow(
            #             eval_points[:, 0],
            #             eval_points[:, 1],
            #             spline_degree,
            #             spline_degree,
            #             ctrl_points,
            #             ctrl_weights,
            #             knots_x,
            #             knots_y,
            #         )
            #     )

            #     hel_rotated = rotate_heliostat(hel_origin, target_hel_coords)
            #     rayPoints = hel_rotated + position_on_field

            #     surface_normals = rotate_heliostat(surface_normals, target_hel_coords)
            #     surface_normals = surface_normals / surface_normals.norm(dim=-1).unsqueeze(-1)
            #     ray_directions = reflect_rays_(from_sun, surface_normals)

            pred_bitmap = R.render()
            if epoch %  10== 0:#
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
        #     if ray_directions.grad is None or (ray_directions.grad == 0).all():
        #         print('no more optimization possible; ending...')
        #         break

        opt.step()
        sched.step(loss)
        # if epoch % 1 == 0:
        #     num_missed = indices.numel() - indices.count_nonzero()
        #     ray_diff = calc_ray_diffs(
        #         ray_directions.detach(),
        #         target_ray_directions,
        #     )
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            # f'missed: {num_missed.detach().cpu().item()}, '
            # f'ray differences: {ray_diff.detach().cpu().item()}'
        )
    # Diff Raytracing >


if __name__ == '__main__':
    main()
