import torch as th
import os

import utils
import numpy as np


import time as to_time
import datetime

# import matplotlib as matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec



def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def test_surfaces(H):

    points_on_hel = H.discrete_points.detach().cpu()
    ideal_vecs = H._normals_ideal.detach().cpu()
    normal_vecs = H.normals.detach().cpu()


    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(ncols=7, figsize=(49,7))
    im1 = ax1.scatter(points_on_hel[:,0],points_on_hel[:,1], c=points_on_hel[:,2])
    im2 = ax2.scatter(points_on_hel[:,0],points_on_hel[:,1], c=ideal_vecs[:,0])
    im3 = ax3.scatter(points_on_hel[:,0],points_on_hel[:,1], c=ideal_vecs[:,1])
    im4 = ax4.scatter(points_on_hel[:,0],points_on_hel[:,1], c=ideal_vecs[:,2])
    im5 = ax5.scatter(points_on_hel[:,0],points_on_hel[:,1], c=normal_vecs[:,0])
    im6 = ax6.scatter(points_on_hel[:,0],points_on_hel[:,1], c=normal_vecs[:,1])
    im7 = ax7.scatter(points_on_hel[:,0],points_on_hel[:,1], c=normal_vecs[:,2])
    plt.show()
    exit()

def plot_surfaces_mrad(heliostat_target, heliostat_pred, epoch, logdir_surfaces, writer = None):
    logdir_mrad = os.path.join(logdir_surfaces, "mrad")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mrad, exist_ok=True)


    target_normal_vecs = heliostat_target.normals
    ideal_normal_vecs = heliostat_target._normals_ideal
    pred_normal_vecs = heliostat_pred.normals
    
    target_angles = th.acos(th.sum(ideal_normal_vecs * target_normal_vecs, dim=-1)).detach().cpu()
    pred_angles = th.acos(th.sum(ideal_normal_vecs * pred_normal_vecs, dim=-1)).detach().cpu()
    
    target_angles = target_angles - th.min(target_angles)
    pred_angles = pred_angles -th.min(pred_angles)

    diff_angles = abs(target_angles-pred_angles)

    if writer:
      writer.add_scalar("test/normal_diffs", th.sum(diff_angles)/len(diff_angles), epoch)

        #Get discrete points
    target_points = heliostat_target.discrete_points
    target_points = target_points.detach().cpu()

    pred_points = heliostat_pred.discrete_points
    pred_points = pred_points.detach().cpu()
    diff_points = pred_points.clone()

    target_points[:,2] = target_angles / 1e-3
    pred_points[:,2] = pred_angles / 1e-3
    diff_points[:,2] = diff_angles / 1e-3
    # print(th.max(pred_angles), th.mean(pred_angles))
    target = target_points 
    pred = pred_points
    # pred = pred[pred[:,-1] >= th.max(target[:,-1])]
    diff = diff_points
    # diff = diff[diff[:,-1] >= th.max(target[:,-1])]


    fig = plt.figure(figsize=(15,6))
    # fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.05])

    # p0 = ax1.get_position().get_points().flatten()
    # p1 = ax2.get_position().get_points().flatten()
    # p2 = ax3.get_position().get_points().flatten()
    # ax_cbar = fig.add_axes([p0[0],  0.05, p1[2]-p0[0], 0.05])
    # ax_cbar1 = fig.add_axes([1.4*p2[0], 0.05, 0.02, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.scatter(target[:,0],target[:,1], c=target[:,2], cmap="magma")
    ax1.set_xlim(th.min(target[:,0]),th.max(target[:,0]))
    ax1.set_ylim(th.min(target[:,1]),th.max(target[:,1]))
    ax1.title.set_text('Original Surface [mrad]')
    ax1.set_aspect("equal")
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.scatter(pred[:,0],pred[:,1], c=pred[:,2], cmap="magma")
    ax2.set_xlim(th.min(pred[:,0]),th.max(pred[:,0]))
    ax2.set_ylim(th.min(pred[:,1]),th.max(pred[:,1]))
    ax2.title.set_text('Predicted Surface [mrad]')
    ax2.set_aspect("equal")
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.scatter(diff[:,0],diff[:,1], c=diff[:,2], cmap="magma")
    ax3.set_xlim(th.min(diff[:,0]),th.max(diff[:,0]))
    ax3.set_ylim(th.min(diff[:,1]),th.max(diff[:,1]))
    ax3.title.set_text('Difference [mrad]')
    ax3.set_aspect("equal")
    ax3.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    
    # ax4 = fig.add_subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1,0:2])
    plt.colorbar(im1, orientation='horizontal', cax=ax4, format='%.0e')
    
    ax5 = plt.subplot(gs[1,2])
    plt.colorbar(im3, orientation='horizontal', cax=ax5, format='%.0e')
    plt.tight_layout()

    fig.savefig(os.path.join(logdir_mrad, f"test_{epoch}"))
    plt.close(fig)


def plot_surfaces_mm(heliostat_target, heliostat_pred, epoch, logdir_surfaces, writer = None):

    logdir_mm = os.path.join(logdir_surfaces, "mm")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mm, exist_ok=True)

    target = heliostat_target.discrete_points
    ideal = heliostat_target._discrete_points_ideal
    target = target.detach().cpu()
    ideal = ideal.detach().cpu()
    
    # print(target.shape)
    # print(ideal.shape)
    target[:,-1] = target[:,-1] - ideal[:,-1]
    
    # target[:,2] = target[:,2]#/1e-3

    pred = heliostat_pred.discrete_points
    pred = pred.detach().cpu()
    pred[:,-1] = pred[:,-1] - ideal[:,-1]

    
    
    
    # pred[:,2] = pred[:,2]#/1e-3

    diff = pred.clone()
    diff[:,2] = pred[:,2]-target[:,2]#/10e-3
    if writer:
        writer.add_scalar("test/location_diffs", th.sum(abs(diff[:,2]))/len(diff[:,2]), epoch)




    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    plt.subplots_adjust(left=0.03, top=0.95, right =0.97, bottom=0.15)

    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0],  0.05, p1[2]-p0[0], 0.05])
    ax_cbar1 = fig.add_axes([p2[0], 0.05, p2[2]-p2[0], 0.05])

    im1 = ax1.scatter(target[:,0],target[:,1], c=target[:,2])
    ax1.set_xlim(th.min(target[:,0]),th.max(target[:,0]))
    ax1.set_ylim(th.min(target[:,1]),th.max(target[:,1]))
    ax1.title.set_text('Original Surface [mm]')
    ax1.set_aspect("equal")

    im2 = ax2.scatter(pred[:,0],pred[:,1], c=pred[:,2])
    ax2.set_xlim(th.min(pred[:,0]),th.max(pred[:,0]))
    ax2.set_ylim(th.min(pred[:,1]),th.max(pred[:,1]))
    ax2.title.set_text('Predicted Surface [mm]')
    ax2.set_aspect("equal")

    im3 = ax3.scatter(diff[:,0],diff[:,1], c=diff[:,2], cmap="magma")
    ax3.set_xlim(th.min(diff[:,0]),th.max(diff[:,0]))
    ax3.set_ylim(th.min(diff[:,1]),th.max(diff[:,1]))
    ax3.title.set_text('Difference [mm]')
    ax3.set_aspect("equal")

    plt.colorbar(im1, cax=ax_cbar, orientation='horizontal', format='%.0e')
    plt.colorbar(im3, cax=ax_cbar1, orientation='horizontal', format='%.0e')

    # colorbar(im3)
    # plt.show()
    # exit()


    fig.savefig(os.path.join(logdir_mm, f"test_{epoch}"))
    plt.close(fig)


def plot_surfaces_3D_mm(heliostat_pred, epoch, logdir, writer = None):
    logdir_mm = os.path.join(logdir, "mm")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir_mm, exist_ok=True)
    
    pred = heliostat_pred.discrete_points
    pred = pred.detach().cpu()
    
    ideal = heliostat_pred._discrete_points_ideal
    ideal = ideal.detach().cpu()
    
    pred[:,2] = pred[:,2]-ideal[:,2]
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_trisurf(pred[:,0], pred[:,1], pred[:,2], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    fig.savefig(os.path.join(logdir_mm, f"test_3D_{epoch}"))
    plt.close(fig)

def plot_surfaces_3D_mrad(heliostat_target, heliostat_pred, epoch, logdir, writer = None):
    logdir_mrad = os.path.join(logdir, "mrad")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir_mrad, exist_ok=True)
    
    points = heliostat_target.discrete_points.detach().cpu()
    
    pred = heliostat_pred.normals
    pred = pred.detach().cpu()
    
    ideal = heliostat_pred._normals_ideal
    ideal = ideal.detach().cpu()

    diff = th.sum(pred * ideal, dim=-1)-0.5115
    
    # pred[:,2] = pred[:,2]-ideal[:,2]
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_zlim(0.00005,0.00015)
    surf = ax.plot_trisurf(points[:,0], points[:,1], diff, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    
    
    # if epoch > 30:
    #     plt.show()
    #     exit()
    # else:
    fig.savefig(os.path.join(logdir_mrad, f"test_3D_{epoch}"))
    plt.close(fig)


def plot_diffs(hel_origin, ideal_normal_vecs, target_normal_vecs, pred_normal_vecs, epoch, logdir):

    # matplotlib.use('Agg')
    differences_target = th.sum(ideal_normal_vecs * target_normal_vecs, dim=-1).detach().cpu()
    differences_pred = th.sum(ideal_normal_vecs * pred_normal_vecs, dim=-1).detach().cpu()

    x = hel_origin[:,0].detach().cpu()
    y = hel_origin[:,1].detach().cpu()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_zlim3d(1, 1.00001)
    surf = ax.plot_trisurf(x, y, abs(differences_target)-abs(differences_pred), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    fig.savefig(os.path.join(logdir, f"test_{epoch}"))
    plt.close(fig)



def plot_normal_vectors(points_on_hel, normal_vectors):
    '''

    Parameters
    ----------
    points_on_hel : (N,3) Tensor
        all points on heliostat.
    normal_vectors : (N,3) Tensor
        direction vector of the corresponding points

    Returns
    -------
    None.

    '''
    matplotlib.use('QT5Agg')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(-4, 4)
    ax.set_zlim3d(0.03, 0.06)
    to = points_on_hel.detach().cpu()
    tv = normal_vectors.detach().cpu()
    # plt.scatter(tv[:,0], tv[:,1], tv[:,2])
    ax.quiver(to[:,0], to[:,1], to[:,2], tv[:,0], tv[:,1], tv[:,2], length=0.01, normalize=False, color="b")
    plt.show()
    exit()


def plot_raytracer(h_rotated, h_matrix, position_on_field, aimpoint,aimpoints, sun):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field
    print(aimpoints.shape)
    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat
    ax.scatter(aimpoint[0],aimpoint[1],aimpoint[2]) #Aimpoint
    ax.scatter(aimpoints[0,:,0],aimpoints[0,:,1],aimpoints[0,:,2])
    ax.scatter(sun[0]*50,sun[1]*50,sun[2]*50) #Sun

    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(0, 50)

    #Heliostat Coordsystem
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    ax.quiver(0, 0, 0, h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    ax.quiver(0, 0, 0, h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    ax.quiver(0, 0, 0, h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    plt.show()
    exit()

# def plot_heliostat(h_rotated, ray_directions):
#     if not isinstance(h_rotated, np.ndarray):
#         h_rotated = h_rotated.detach().cpu().numpy()
#     if (
#             ray_directions is not None
#             and not isinstance(ray_directions, np.ndarray)
#     ):
#         ray_directions = ray_directions.detach().cpu().numpy()

#     fig = plt.figure()
#     ax = plt.axes(projection='3d')


#     if h_rotated.ndim == 3:
#         h_rotated = h_rotated.reshape(-1, h_rotated.shape[-1])
#     # aimpoints = aimpoints-position_on_field
#     # aimpoint = aimpoint-position_on_field

#     ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat

#     ax.set_xlim3d(-50, 0)
#     ax.set_ylim3d(-10, 10)
#     ax.set_zlim3d(0, 5)
#     if ray_directions is not None:
#         ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[:,0], ray_directions[:,1], ray_directions[:,2], length=50, normalize=True, color="b")
#         # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[1][0], ray_directions[1][1], ray_directions[1][2], length=1, normalize=True, color="g")
#         # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[2][0], ray_directions[2][1], ray_directions[2][2], length=1, normalize=True, color="r")
#     plt.show()
#     exit()


def target_image_comparision_pred_orig_naive(ae, 
                                             original, 
                                             predicted, 
                                             naive,
                                             train_sun_position, 
                                             epoch,
                                             logdir,
                                             start_main_plot_at_row =1
                                             ):
    
    num_azi = len(th.unique(ae[:,0]))
    num_ele = len(th.unique(ae[:,1]))
    
    ae = ae.detach().cpu()
    train_sun_position = train_sun_position.detach().cpu()
    
    small_width=[0.2]*num_ele*4
    width_ratios= [1]*num_ele *4
    width_ratios[3::4]=small_width[3::4]

    column = num_azi
    row    = num_ele
    
    height_ratios   =[1]*(num_azi+start_main_plot_at_row)
    height_ratios[0]=2
    

    
    
    loss =th.nn.L1Loss()
    row = num_ele *4
    column = num_azi
    fig, axs = plt.subplots(column+1, row, figsize=(5*num_ele, 2*num_azi), sharex=True, sharey=True, 
                            gridspec_kw={'width_ratios': width_ratios,'height_ratios':height_ratios})
    gs = axs[1, 1].get_gridspec()
    
    j=0
    
    original = original.detach().cpu()
    predicted = predicted.detach().cpu()
    naive = naive.detach().cpu()
    
    
    smp = start_main_plot_at_row*row#start main plot 
    for i, ax in enumerate(axs.flat):
        #Nested Subplots
        # if i==0:
        #     ax.remove()
        
        if i<smp:
            ax.remove()
        #Modifications for all Plots
        if i>=smp:
            ax.set_aspect('equal')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        
        #Modification for each row
            
        if i%row==0 and i>=smp:
            ax.set_ylabel("Azimuth = "+str(int(ae[j,0])), fontweight='bold', fontsize=12)
    
        #     ax.set_ylabel("Azimuth = "+str(int(ae[j,0])))
        
        #Modification specific for each plot
        
        output_pred = loss(predicted[j],original[j])
        output_naive = loss(naive[j],original[j])
        
        if i%4==0 and i>=smp:
            
            ax.imshow(predicted[j], cmap = "coolwarm")

            if output_pred < output_naive:
                font = "bold"
            else:
                font = "normal"
            ax.set_xlabel("L1: "+f"{output_pred.item():.4f}", fontweight=font)
            if i-smp<row:
                ax.set_title('Predicted', fontweight='bold')
            
        elif i%4==1 and i>=smp:
            ax.imshow(original[j], cmap = "coolwarm")
            if i-smp<row:
                ax.set_title('Original', fontweight='bold')
            if i-smp>=row*(column-1):
                ax.set_xlabel("Elevation = "+str(int(ae[j,1])), fontweight='bold', fontsize=12)

                
        elif i%4==2 and i>=smp:
            ax.imshow(naive[j], cmap ="coolwarm")
            
            if output_naive < output_pred:
                font = "bold"
            else:
                font = "normal"
            
            ax.set_xlabel("L1: "+f"{output_naive.item():.4f}",fontweight=font)
            if i-smp<row: 
                ax.set_title('Naive', fontweight='bold')
        elif i%4==3 and i>=smp:
            ax.remove()
    
            
        if i%4==3 and i>=smp:
            j+=1
            
    axbig = fig.add_subplot(gs[0:1, 4:8],projection='polar')

    axbig.set_thetamin(-90)
       
    axbig.set_thetamax(90)
       
    axbig.set_theta_zero_location("N")
       
    axbig.set_rorigin(-95)
    
    axbig.scatter(th.deg2rad(ae[:,0]), -ae[:,1],color = 'r',marker='x',s=10, label="Test sun positions")
    
    train_sun_position= utils.vec_to_ae(train_sun_position)
    axbig.scatter(th.deg2rad(train_sun_position[:,0]),-train_sun_position[:,1] ,color = 'b',marker='x',s=10, label="Train sun position")
    axbig.legend(loc='upper right',bbox_to_anchor=(-0.1, 0.5, 0.5, 0.5))
    axbig.set_yticks(th.arange(-90, 20, 30))
    
    
    axbig.set_yticklabels(abs(axbig.get_yticks()))
    axbig.set_ylabel('Azimuth',rotation=67.5)
    axbig.set_xlabel('Elevation')
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"enhanced_test_{epoch}"))
    plt.close(fig)
    
def spherical_loss_plot(train_vec, spheric_ae, train_loss, spheric_losses, naive_losses, num_spheric_samples, epoch, logdir):
    """
    spheric_ae and losses are seperated in 3 parts
    [:nums] = constant elevation and western azimuth hemisphere (from train vecor viewing position)
    [nums:2*nums] = constant elevation and eastern hemisphere
    [2*nums:] = constant azimuth with all possible elevations
    """
    
    #To CPU
    train_vec = train_vec.detach().cpu()
    train_loss = train_loss.detach().cpu()
    spheric_losses = spheric_losses.detach().cpu()
    naive_losses = naive_losses.detach().cpu()
    #Radial Plot Calculations
    train_ae = utils.vec_to_ae(train_vec)
    ae = spheric_ae.clone()

    #Setup Figure
    height_ratios=[1, 0.3]
    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=height_ratios)
    
    #Fill First Plot
    ax1 = fig.add_subplot(gs[0, 0],projection='polar')
    ax1.set_theta_zero_location("N")
    
    nums =num_spheric_samples
    l1 = ax1.scatter(th.deg2rad(ae[:nums,0]), -ae[:nums,1],marker='.',s=40, label="Test: Left-handed azimuth angles")
    l2 = ax1.scatter(th.deg2rad(ae[nums:2*nums,0]), -ae[nums:2*nums,1],marker='.',s=40, label="Test: Right-handed azimuth angles")
    l3 = ax1.scatter(th.deg2rad(ae[2*nums:,0]), -ae[2*nums:,1],marker='.',s=40, label="Test: Elevation angles")
    l4 = ax1.scatter(th.deg2rad(train_ae[:,0]), -train_ae[:,1],marker='*',s=40, label="Azi./Ele. of training")
    
    #Axis Ticks
    ax1.set_yticks(th.arange(-90, -10, 20))
    ax1.set_yticklabels(abs(ax1.get_yticks()))
    ax1.set_rlabel_position(0)
    tick_labels = ["0","45","90","135","$\pm$ 180","-135","-90","-45"]
    value_list = ax1.get_xticks().tolist()
    ax1.xaxis.set_ticks(value_list)
    ax1.set_xticklabels(tick_labels)
    #Axis Labels
    ax1.set_xlabel(r'Azimuth $\theta^{a}$ [°]')
    label_position=ax1.get_rlabel_position()
    ax1.text(np.deg2rad(label_position+7),-63,r'Elevation $\theta^{e}$[°]',
        rotation= 91,ha='center',va='center')

    #Calculations For Second Plot
    #predictions
    ae[:nums,0] = ae[:nums,0]-train_ae[0,0]
    azi_west_no_offsets = th.abs(th.where(ae[:nums,0]>0, ae[:nums,0], 360+ae[:nums,0])-360)#delay to zero
    azi_west_loss = spheric_losses[:nums] # same for y values
    
    ae[nums:2*nums,0] = ae[nums:2*nums,0]-train_ae[0,0]
    azi_east_no_offsets = th.where(ae[nums:2*nums,0]>0, ae[nums:2*nums,0], 360+ae[nums:2*nums,0])%360#delay to 0-180
    azi_east_loss = spheric_losses[nums:2*nums] # same for y values
    
    ele_no_offsets = th.where(ae[2*nums:,0]<0, ae[2*nums:,1]-train_ae[0,1], 180-ae[2*nums:,1]-train_ae[0,1])#delay to 180-0
    ele_loss = spheric_losses[2*nums:]
    
    #naive
    naive_azi_west_loss   = naive_losses[:nums] # same for y values
    naive_azi_east_loss   = naive_losses[nums:2*nums] # same for y values
    naive_ele_loss        = naive_losses[2*nums:]
    
    
    
    
    #Fill Second Figure
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.plot(azi_west_no_offsets,azi_west_loss,marker='.',zorder=3)
    ax2.plot(azi_east_no_offsets,azi_east_loss,marker='.',zorder=6)
    ax2.plot(ele_no_offsets,ele_loss,marker='.',zorder=8)
    ax2.scatter(0,train_loss,s=90,marker='*', color="r",zorder=10)
    
    l1_naive = ax2.plot(azi_west_no_offsets,naive_azi_west_loss,zorder=3, color="cornflowerblue", label="Naive loss left-azi")
    l2_naive = ax2.plot(azi_east_no_offsets,naive_azi_east_loss,zorder=6, color="bisque", label="Naive loss right-azi")
    l3_naive = ax2.plot(ele_no_offsets,naive_ele_loss,zorder=8, color="limegreen", label="Naive loss ele")
    
    # ax2.set_xlim(-20,180)
    
    #Axis Labels
    ax2.set_xlabel(r'$|\theta^{a,e}_{test}|-\theta^{a,e}_{train}$ [°]')
    ax2.set_ylabel('L1 Loss')
    # ax2.set_ylim(0,12)

    #Legend
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.legend(handles=[l1,l1_naive[0],l2,l2_naive[0],l3,l3_naive[0], l4], loc="center")
    
    
    # plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"spheric_test_{epoch}"))
    plt.close(fig)


def season_plot(season_extras, ideal, prediction, ground_truth, prediction_loss,ground_truth_loss, logdir, epoch):
    # import matplotlib as matplotlib

    # from matplotlib.backends.backend_pgf import FigureCanvasPgf
    # matplotlib.backend_bases.register_backend('png', FigureCanvasPgf)
    # from matplotlib import pyplot as plt
    # matplotlib.use('pgf')
    # plt.rcParams.update({
    #     'pgf.texsystem': 'pdflatex',
    #     'pgf.preamble': r'\usepackage[utf8]{inputenc}\usepackage{color}\usepackage{dashrule}\usepackage{amssymb}\usepackage{stackengine}\usepackage{scalerel}\usepackage{xcolor}\usepackage{graphicx}',
    #     'text.usetex': True,
    #     'text.latex.preamble':  r'\usepackage{color}\usepackage{dashrule}',
    # })
    # pgf_with_latex = {
    #     "text.usetex": True,            # use LaTeX to write all text
    #     "pgf.rcfonts": False,           # Ignore Matplotlibrc
    #     "pgf.preamble": [
    #         r'\usepackage{color}'     # xcolor for colours
    #     ]
    # }
    # matplotlib.rcParams.update(pgf_with_latex)
    #To CPU
    season_extras = season_extras.copy()
    date_time_ae = season_extras.pop("date_time_ae")
    # print(len(date_time_ae))
    tmp = []
    for i in range(len(date_time_ae)):
        tmp.append(th.tensor(date_time_ae[i]))
    date_time_ae = th.cat(tmp).squeeze()
    # season_extras.pop('date_time_ae', None)
    # print(date_time_ae)
    # date_time_ae = date_time_ae.detach().cpu()
    ideal = ideal.detach().cpu()
    prediction = prediction.detach().cpu()
    ground_truth = ground_truth.detach().cpu()
    prediction_loss = prediction_loss.detach().cpu()
    ground_truth_loss = ground_truth_loss.detach().cpu()
    
    image_size= prediction.shape[-1]
    # print(image_size)
    
    prediction_norm = th.linalg.norm(prediction, dim=(1,2))
    prediction_loss = prediction_loss/prediction_norm

    # exit()

    dt = date_time_ae[:,0:6]
    # print(len(dt))
    # exit()
    
    d = [datetime.datetime(int(dt[x,0]),
                           int(dt[x,1]),
                           int(dt[x,2]),
                           int(dt[x,3]),
                           int(dt[x,4]),
                           int(dt[x,5])
                           ) for x in range(len(dt))]
    # print(len(d))
    unixtime = th.tensor([to_time.mktime(d[x].timetuple()) for x in range(len(d))])
    ae = date_time_ae[:,6:8]
    # print(ae[0,0:2])

#Setup Figure
    # height_ratios= [1, 0.3]
    grid_width =24
    grid_height = 6
    
    small_width=[0.2]*grid_width
    mid_width=[0.4]*grid_width
    width_ratios= [1]*grid_width
    width_ratios[3::4]=small_width[3::4]
    width_ratios[2] = 0.1
    width_ratios[19:] = mid_width[19:]
    
    height_ratios = [1]*grid_height
    height_ratios[0]=0.0001
    # height_ratios[-1]=0.0001
    
    fig = plt.figure(constrained_layout=True, figsize=(27,10))
    
    gs = GridSpec(grid_height, grid_width, figure=fig, width_ratios=width_ratios,height_ratios=height_ratios, wspace=0)#height_ratios=height_ratios
    
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    
#Headlines
    ax = fig.add_subplot(gs[1, 0:3])
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(" ")
    ax.set_title("Seasonal Trajectories")
    ax.axis('off')
    
#Fill First Plot
    ax_polar = fig.add_subplot(gs[1:4, 0:3],projection='polar') #####XXXXXXXXXXXXXX
    ax_polar.set_theta_zero_location("N")
    
    index = 0
    legend = []
    seasons = [None,None,None,None]
    for key in season_extras:
        # print(key)
        if not key == "date_time_ae":
            # print(index,new_index)
            new_index = index + season_extras[key]
        
            ae_season = ae[index:new_index]
            dt_season = dt[index:new_index]
            unix_season = unixtime[index:new_index]
            prediction_season =  prediction[index:new_index]
            ground_truth_season = ground_truth[index:new_index]
            ideal_season = ideal[index:new_index]
            
            all_infos = [prediction_season, ground_truth_season, ideal_season, ae_season, dt_season, unix_season]
            
            highlighted_6  = ae_season[th.logical_and(dt_season[:,3] == 6.0  ,dt_season[:,4] == 0.0)]
            highlighted_9  = ae_season[th.logical_and(dt_season[:,3] == 9.0  ,dt_season[:,4] == 0.0)]
            highlighted_12 = ae_season[th.logical_and(dt_season[:,3] == 12.0 ,dt_season[:,4] == 0.0)]
            highlighted_15 = ae_season[th.logical_and(dt_season[:,3] == 15.0 ,dt_season[:,4] == 0.0)]
            
            unicode = '\\bigstar'
            if key == "measurement":
                label = "Day of measurement 28.10.2021"
                
                color = "orange"
                # short_label = fr'$\textcolor{{{color}}}{{{unicode}}}$ Measurement'
                short_label = "Measurement"
                seasons[0] = all_infos
                ax = fig.add_subplot(gs[1, 3])
                ver_pos=0.2
                
            
            if key == "short":
                label="Solstice (shortest day) 21.12.2021"
                color = "blue"
                # short_label = fr'$\textcolor{{{color}}}{{{unicode}}}$ Shortest day'
                short_label = "Shortest day"
                
                seasons[1] = all_infos
                
                ax = fig.add_subplot(gs[2, 3])
                ver_pos =0.2
            
            if key == "spring":
                label="Equinox 20.03.2022"
                color = "green"
                # short_label = fr'$\textcolor{{{color}}}{{{unicode}}}$ Equinox'
                short_label = "Equinox"
                
                seasons[2] = all_infos
                
                ax = fig.add_subplot(gs[3, 3])
                ver_pos = 0.3
            
            if key == "long":
                label = "Solstice (longest day) 21.06.2022"
                color = "red"
                unicode = '\\bigstar'
                # short_label = fr'$\textcolor{{{color}}}{{{unicode}}}$ Longest day'
                short_label = "Longest day"
    
                seasons[3] = all_infos
                ax = fig.add_subplot(gs[4, 3])
                ver_pos = 0.2
                
            
            ax.text(0.5,ver_pos, short_label, rotation=90)
            ax.set_axis_off()
            
            
            
            l = ax_polar.scatter(-th.deg2rad(180-ae_season[:,0]), -ae_season[:,1],marker='.',s=40, color = color, label=label)
            ax_polar.scatter(-th.deg2rad(180-highlighted_6[:,0]), -highlighted_6[:,1],marker='*',s=40, color = color, label=label)
            ax_polar.scatter(-th.deg2rad(180-highlighted_9[:,0]), -highlighted_9[:,1],marker='*',s=40, color = color, label=label)
            ax_polar.scatter(-th.deg2rad(180-highlighted_12[:,0]), -highlighted_12[:,1],marker='*',s=40, color = color, label=label)
            ax_polar.scatter(-th.deg2rad(180-highlighted_15[:,0]), -highlighted_15[:,1],marker='*',s=40, color = color, label=label)
            legend.append(l)
            index = new_index
    
    l_star = Line2D([0], [0], marker='*', label='Selected images', markersize=7, linewidth=0, color="black")
    legend.append(l_star)
#Axis Ticks
    ax_polar.set_yticks(th.arange(-90, -0, 15))
    ax_polar.set_yticklabels(abs(ax_polar.get_yticks()))
    ax_polar.set_rlabel_position(0)
    tick_labels = ["0","45","90","135","$\pm$ 180","-135","-90","-45"]
    value_list = ax_polar.get_xticks().tolist()
    ax_polar.xaxis.set_ticks(value_list)
    ax_polar.set_xticklabels(tick_labels)
    # # #Axis Labels
    ax_polar.set_xlabel(r'Azimuth $\theta^{a}$ [°]')
    label_position=ax_polar.get_rlabel_position()
    ax_polar.text(np.deg2rad(label_position+7),-45,r'Elevation $\theta^{e}$[°]',
        rotation= 91,ha='center',va='center')
    


#Calculations For Second Plot
    i = 1
    j = 0
    loss =th.nn.L1Loss()
    for season in seasons:
        if season:

            
            prediction = season[0]
            ground_truth = season[1]
            ideal = season[2]
            time_season = season[4]
            
            prediction_list = []
            prediction_list.append(prediction[th.logical_and(time_season[:,3] == 15.0  ,time_season[:,4] == 0.0)].squeeze())
            prediction_list.append(prediction[th.logical_and(time_season[:,3] == 12.0  ,time_season[:,4] == 0.0)].squeeze())
            prediction_list.append(prediction[th.logical_and(time_season[:,3] == 9.0 ,time_season[:,4] == 0.0)].squeeze())
            prediction_list.append(prediction[th.logical_and(time_season[:,3] == 6.0 ,time_season[:,4] == 0.0)].squeeze())
            
            ground_truth_list = []
            ground_truth_list.append(ground_truth[th.logical_and(time_season[:,3] == 15.0  ,time_season[:,4] == 0.0)].squeeze())
            ground_truth_list.append(ground_truth[th.logical_and(time_season[:,3] == 12.0  ,time_season[:,4] == 0.0)].squeeze())
            ground_truth_list.append(ground_truth[th.logical_and(time_season[:,3] == 9.0 ,time_season[:,4] == 0.0)].squeeze())
            ground_truth_list.append(ground_truth[th.logical_and(time_season[:,3] == 6.0 ,time_season[:,4] == 0.0)].squeeze())
            
            ideal_list =[]
            ideal_list.append(ideal[th.logical_and(time_season[:,3] == 15.0  ,time_season[:,4] == 0.0)].squeeze())
            ideal_list.append(ideal[th.logical_and(time_season[:,3] == 12.0  ,time_season[:,4] == 0.0)].squeeze())
            ideal_list.append(ideal[th.logical_and(time_season[:,3] == 9.0 ,time_season[:,4] == 0.0)].squeeze())
            ideal_list.append(ideal[th.logical_and(time_season[:,3] == 6.0 ,time_season[:,4] == 0.0)].squeeze())
            
            output_pred = loss(prediction[j],ground_truth[j])
            output_naive = loss(ideal[j],ground_truth[j])
            
#Fill Second Plot    



            for k in range(0,16):
                
                if i==1 and j==3:
                    ax_title = fig.add_subplot(gs[i, k+4])
                    
                    ax_title.xaxis.set_label_position('top')
                    if k%4==0:
                        ax_title.set_xlabel('Ground Truth')
                    elif k%4==1:
                        ax_title.set_title('6:00 a.m.')
                        ax_title.set_xlabel("Ideal")
                    elif k%4==2:
                        ax_title.set_xlabel("Prediction")
                    ax_title.spines['left'].set_visible(False)
                    ax_title.spines['right'].set_visible(False)
                    ax_title.spines['bottom'].set_visible(False)
                    ax_title.spines['top'].set_visible(False)
                    ax_title.axes.yaxis.set_ticklabels([])
                    ax_title.axes.xaxis.set_ticklabels([])
                    plt.tick_params(left=False,
                        right=False,top=False,bottom=False)
                        
                        
                    # ax_title.axis('off')
                    # ax_title.get_xaxis().set_visible(False)
                    # ax_title.get_yaxis().set_visible(False)
                # print(i,j,k)
                
                pred = prediction_list[j]

                gt = ground_truth_list[j]
                ide = ideal_list[j]

                if k%4==0:
                    if not len(pred) == 0:
                        ax = fig.add_subplot(gs[i, k+4])
                        ax.imshow(gt, cmap = "coolwarm")
                        if i == 1 and not j==3:
                            ax.xaxis.set_label_position('top')
                            ax.set_xlabel('Ground Truth')



                elif k%4==1:
                    if not len(gt) == 0:
                        ax = fig.add_subplot(gs[i, k+4])
                        ax.imshow(ide, cmap = "coolwarm")
                        
                        if output_naive < output_pred:
                            font = "bold"
                        else:
                            font = "normal"
                        ax.text(0,image_size+0.1*image_size, "L1: "+f"{output_naive.item():.4f}", fontweight=font, va="bottom",size="medium")
                        
                        if i == 1:
                            
                            ax.xaxis.set_label_position('top')
                            ax.set_xlabel('Ideal')
                            
                            
                        
                elif k%4==2:
                    if not len(ide) == 0:
                        ax = fig.add_subplot(gs[i, k+4])
                        ax.imshow(pred, cmap ="coolwarm")
                            
                        if output_pred < output_naive :
                            font = "bold"
                        else:
                            font = "normal"
                        
                        ax.text(0,image_size+0.1*image_size,"L1: "+f"{output_pred.item():.4f}",fontweight=font, va="bottom", size="medium")
                        
                        if i == 1:
                            ax.xaxis.set_label_position('top')
                            ax.set_xlabel('Prediction')
                
                ax.set_aspect('equal')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
                
                if k%4 == 1 and i==1:
                    if j ==0:
                        ax.set_title('03:00 p.m.')
                    elif j==1:
                        ax.set_title('12:00 p.m.')
                    elif j==2:
                        ax.set_title('9:00 a.m.')
                

                

                if k%4==3:
                    j+=1
            i+=1
        j=0

#Calculations third plot
    stepwidth = unixtime[1]-unixtime[0]
    sorted_prediction_loss = th.tensor([x for _, x in sorted(zip(d,prediction_loss))])
    # sorted_prediction = th.tensor([x for _, x in sorted(zip(d,prediction))])
    # print(th.linalg.norm(sorted_prediction))
    sorted_d = sorted(d)
    # print(len(sorted_d), sorted_prediction.shape)
    sorted_unix = [x for _, x in sorted(zip(d,unixtime))]

    new_axis_index =[i for i in range(len(sorted_unix)-1) if sorted_unix[i+1]-sorted_unix[i] >(1.2*stepwidth)]
    new_axis_index.append(len(sorted_prediction_loss))
    
    old_split = 0
    position_start = 19
    position_end = 20
    
#Fill thirs plot
    for i in range(len(new_axis_index)):
        split = new_axis_index[i]+1
        # print(start,end)
        ax = fig.add_subplot(gs[1:5, position_start:position_end])
        myFmt = mdates.DateFormatter('%y-%m-%d %H') # here you can format your datetick labels as desired
        plt.gca().xaxis.set_major_formatter(myFmt)

        position_start +=1
        position_end+=1

        ax.plot_date(sorted_d, sorted_prediction_loss)
        
        min_tick = min(sorted_d[old_split:split])
        # min_value -= datetime.timedelta(hours=1)
        max_tick = max(sorted_d[old_split:split])
        ax.set_xticks([min_tick,max_tick])
        
        min_value = min_tick - datetime.timedelta(hours=1)
        max_value = max_tick + datetime.timedelta(hours=1)
        # print(min_value,max_value)
        old_split = split
        ax.set_xlim(min_value, max_value)  # outliers only
        
        plt.xticks(rotation=75, ha="right")
        line = .01  # how big to make the diagonal lines in axes coordinates
        if i==0:
            ax.spines['right'].set_visible(False)
            
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-line,1+line), (-line,+line), **kwargs)
            ax.plot((1-line,1+line),(1-line,1+line), **kwargs)
            ax.set_ylabel(r"Image-wise normed loss $ \frac{L1}{||P||_{Frobenius}}$", size="large")
            # kwargs.update(transform=ax.transAxes)  # switch to the bottom axes

            
        elif i==(len(new_axis_index)-1):
            ax.spines['left'].set_visible(False)
            ax.axes.yaxis.set_ticklabels([])
            plt.tick_params(left=False,
                )
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            
            kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
            ax.plot((-line,+line), (1-line,1+line), **kwargs)
            ax.plot((-line,+line), (-line,+line), **kwargs)
            

        else:
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tick_params(left=False,
                right=False)
            
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-line,1+line), (-line,+line), **kwargs)
            ax.plot((1-line,1+line),(1-line,1+line), **kwargs)
            
            kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
            ax.plot((-line,+line), (1-line,1+line), **kwargs)
            ax.plot((-line,+line), (-line,+line), **kwargs)
            ax.axes.yaxis.set_ticklabels([])
            
    ax = fig.add_subplot(gs[4, 19:23])
    ax.text(0.5,-0.85,"Time [h]",ha="center")
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 19:23])
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(' ')
    ax.set_title('Time Evolution')
    ax.axis('off')
    

    #Legend
    ax_l = fig.add_subplot(gs[4, 0:2])
    ax_l.axis('off')
    ax_l.legend(handles=legend, loc="right")

    plt.savefig(os.path.join(logdir, f"season_test_{epoch}"), dpi=fig.dpi)
    plt.close(fig)


