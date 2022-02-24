import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import torch as th
import os
# import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils

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
    pred = pred[pred[:,-1] <= th.max(target[:,-1])]
    diff = diff_points
    diff = diff[diff[:,-1] <= th.max(target[:,-1])]


    fig = plt.figure(figsize=(15,6))
    # fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.05])

    # p0 = ax1.get_position().get_points().flatten()
    # p1 = ax2.get_position().get_points().flatten()
    # p2 = ax3.get_position().get_points().flatten()
    # ax_cbar = fig.add_axes([p0[0],  0.05, p1[2]-p0[0], 0.05])
    # ax_cbar1 = fig.add_axes([1.4*p2[0], 0.05, 0.02, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.scatter(target[:,0],target[:,1], c=target[:,2])
    ax1.set_xlim(th.min(target[:,0]),th.max(target[:,0]))
    ax1.set_ylim(th.min(target[:,1]),th.max(target[:,1]))
    ax1.title.set_text('Original Surface [mrad]')
    ax1.set_aspect("equal")
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.scatter(pred[:,0],pred[:,1], c=pred[:,2])
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
    ideal = heliostat_target.ideal_discrete_points
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
    
    ideal = heliostat_pred.ideal_discrete_points
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
    
    print(pred.shape, ideal.shape, (pred*ideal).shape)
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
    ax1.text(th.math.radians(label_position+7),-63,r'Elevation $\theta^{e}$[°]',
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

#Not Used Anymore but not deleted
# if epoch %  10== 0:#
#     im.set_data(pred.detach().cpu().numpy())
#     im.autoscale()
#     plt.savefig(os.path.join("images", f"{epoch}.png"))
