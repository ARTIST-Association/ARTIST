import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torch as th
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_surfaces_mrad(heliostat_target, heliostat_pred, epoch, logdir_surfaces, writer = None):
    logdir_mrad = os.path.join(logdir_surfaces, "mrad")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mrad, exist_ok=True)

    
    target_normal_vecs = heliostat_target.normals
    ideal_normal_vecs = heliostat_target._normals_ideal
    pred_normal_vecs = heliostat_pred.normals

    target_angles = th.sum(ideal_normal_vecs * target_normal_vecs, dim=-1).detach().cpu().numpy()
    pred_angles = th.sum(ideal_normal_vecs * pred_normal_vecs, dim=-1).detach().cpu().numpy()
    diff_angles = abs(target_angles-pred_angles)
    
    if writer:
      writer.add_scalar("test/normal_diffs", np.sum(diff_angles)/len(diff_angles), epoch)
    
        #Get discrete points
    target_points = heliostat_target.discrete_points
    target_points = target_points.detach().cpu().numpy()
    
    pred_points = heliostat_pred.discrete_points
    pred_points = pred_points.detach().cpu().numpy()
    diff_points = pred_points.copy()
    
    target_points[:,2] = target_angles #/ 1e-3
    pred_points[:,2] = pred_angles #/ 1e-3
    diff_points[:,2] = diff_angles #/ 1e-3
    
    target = target_points 
    pred = pred_points
    diff = diff_points 



    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    plt.subplots_adjust(left=0.03, top=0.95, right =0.97, bottom=0.15)
    
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0],  0.05, p1[2]-p0[0], 0.05])
    ax_cbar1 = fig.add_axes([p2[0], 0.05, p2[2]-p2[0], 0.05])
    
    im1 = ax1.scatter(target[:,0],target[:,1], c=target[:,2])
    ax1.set_xlim(np.min(target[:,0]),np.max(target[:,0]))
    ax1.set_ylim(np.min(target[:,1]),np.max(target[:,1]))
    ax1.title.set_text('Original Surface [mrad]')
    ax1.set_aspect("equal")
    
    im2 = ax2.scatter(pred[:,0],pred[:,1], c=pred[:,2])
    ax2.set_xlim(np.min(pred[:,0]),np.max(pred[:,0]))
    ax2.set_ylim(np.min(pred[:,1]),np.max(pred[:,1]))
    ax2.title.set_text('Predicted Surface [mrad]')
    ax2.set_aspect("equal")
    
    im3 = ax3.scatter(diff[:,0],diff[:,1], c=diff[:,2], cmap="magma")
    ax3.set_xlim(np.min(diff[:,0]),np.max(diff[:,0]))
    ax3.set_ylim(np.min(diff[:,1]),np.max(diff[:,1]))
    ax3.title.set_text('Difference [mrad]')
    ax3.set_aspect("equal")
    
    plt.colorbar(im1, cax=ax_cbar, orientation='horizontal', format='%.0e')
    plt.colorbar(im3, cax=ax_cbar1, orientation='horizontal', format='%.0e')
    
    fig.savefig(f"{logdir_mrad}//test_{epoch}")


def plot_surfaces_mm(heliostat_target, heliostat_pred, epoch, logdir_surfaces, writer = None):
    
    logdir_mm = os.path.join(logdir_surfaces, "mm")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mm, exist_ok=True)

    target = heliostat_target.discrete_points
    target = target.detach().cpu().numpy()
    target[:,2] = target[:,2]#/1e-3
    
    pred = heliostat_pred.discrete_points
    pred = pred.detach().cpu().numpy()
    pred[:,2] = pred[:,2]#/1e-3
    
    diff = pred.copy()
    diff[:,2] = pred[:,2]-target[:,2]#/10e-3
    if writer:
        writer.add_scalar("test/location_diffs", np.sum(abs(diff[:,2]))/len(diff[:,2]), epoch)
    
    


    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    plt.subplots_adjust(left=0.03, top=0.95, right =0.97, bottom=0.15)
    
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0],  0.05, p1[2]-p0[0], 0.05])
    ax_cbar1 = fig.add_axes([p2[0], 0.05, p2[2]-p2[0], 0.05])
    
    im1 = ax1.scatter(target[:,0],target[:,1], c=target[:,2])
    ax1.set_xlim(np.min(target[:,0]),np.max(target[:,0]))
    ax1.set_ylim(np.min(target[:,1]),np.max(target[:,1]))
    ax1.title.set_text('Original Surface [mm]')
    ax1.set_aspect("equal")
    
    im2 = ax2.scatter(pred[:,0],pred[:,1], c=pred[:,2])
    ax2.set_xlim(np.min(pred[:,0]),np.max(pred[:,0]))
    ax2.set_ylim(np.min(pred[:,1]),np.max(pred[:,1]))
    ax2.title.set_text('Predicted Surface [mm]')
    ax2.set_aspect("equal")
    
    im3 = ax3.scatter(diff[:,0],diff[:,1], c=diff[:,2], cmap="magma")
    ax3.set_xlim(np.min(diff[:,0]),np.max(diff[:,0]))
    ax3.set_ylim(np.min(diff[:,1]),np.max(diff[:,1]))
    ax3.title.set_text('Difference [mm]')
    ax3.set_aspect("equal")
    
    plt.colorbar(im1, cax=ax_cbar, orientation='horizontal', format='%.0e')
    plt.colorbar(im3, cax=ax_cbar1, orientation='horizontal', format='%.0e')
    
    # colorbar(im3)
   
    
    
    fig.savefig(f"{logdir_mm}//test_{epoch}")


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

    fig.savefig(f"{logdir}//test_{epoch}")



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
    ax.set_zlim3d(0, 2)
    to = points_on_hel.detach().cpu()
    tv = normal_vectors.detach().cpu()
    ax.quiver(to[:,0], to[:,1], to[:,2], tv[:,0], tv[:,1], tv[:,2], length=0.1, normalize=False, color="b")
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

def plot_heliostat(h_rotated, ray_directions):
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    if h_rotated.ndim == 3:
        h_rotated = h_rotated.reshape(-1, h_rotated.shape[-1])
    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field

    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat

    ax.set_xlim3d(-50, 0)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(0, 5)
    if ray_directions is not None:
        ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[:,0], ray_directions[:,1], ray_directions[:,2], length=50, normalize=True, color="b")
        # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[1][0], ray_directions[1][1], ray_directions[1][2], length=1, normalize=True, color="g")
        # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[2][0], ray_directions[2][1], ray_directions[2][2], length=1, normalize=True, color="r")
    plt.show()
    exit()


#Not Used Anymore but not deleted
# if epoch %  10== 0:#
#     im.set_data(pred.detach().cpu().numpy())
#     im.autoscale()
#     plt.savefig(os.path.join("images", f"{epoch}.png"))
