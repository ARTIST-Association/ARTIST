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



def plot_surfaces(hel_origin, ideal_normal_vecs, target_normal_vecs, pred_normal_vecs, epoch, logdir, writer = None):


    
    target = th.sum(ideal_normal_vecs * target_normal_vecs, dim=-1).detach().cpu().numpy()
    pred = th.sum(ideal_normal_vecs * pred_normal_vecs, dim=-1).detach().cpu().numpy()
    diff = abs(pred-target)
    
    
    im_target = target.reshape(int(np.sqrt(len(target))),int(np.sqrt(len(target))))
    im_pred = pred.reshape(int(np.sqrt(len(target))),int(np.sqrt(len(target))))
    im_diff = diff.reshape(int(np.sqrt(len(target))),int(np.sqrt(len(target)))) 
    
    
    if writer:
      writer.add_scalar("train/normal_diffs", np.sum(diff)/len(diff), epoch)

    
    minmin = np.min((np.min(target), np.min(pred)))
    maxmax = np.max((np.max(target), np.max(pred)))
    matplotlib.use('Agg')
    plt.close("all")
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    
    im1 = ax1.imshow(im_target, cmap="plasma", vmin=minmin, vmax=maxmax)#
    im2 = ax2.imshow(im_pred, cmap="plasma", vmin=minmin, vmax=maxmax)#
    colorbar(im1)
    
    im3 = ax3.imshow(im_diff, cmap='jet', norm=matplotlib.colors.LogNorm())
    colorbar(im3)
   
    
    plt.tight_layout(h_pad=0.5)
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])

    # fig.colorbar(im1, ax = axes[3])
    # fig.colorbar(im3, ax = axes[4])
    
    # bbox_ax = axes[0].get_position()
    
    # fig.add_axes() adds the colorbar axes
    # they're bounded by [x0, y0, x_width, y_width]

    # hel_origin = hel_origin.detach().cpu()

    # x = hel_origin[:,0].detach().cpu()
    # y = hel_origin[:,1].detach().cpu()
    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    # # ax.set_zlim3d(1, 1.00001)
    
    # surf = ax.imshow(x, y, abs(differences_target), cmap=cm.coolwarm,
    #                         linewidth=0, antialiased=False)
    # # surf = ax.scatter(hel_origin[:,0],hel_origin[:,1],hel_origin[:,2])
    # surf = ax.plot_trisurf(x, y, differences_pred, cmap=cm.coolwarm,
    #                         linewidth=0, antialiased=False)
    # # fig.show()
    # exit()
    fig.savefig(f"{logdir}//test_{epoch}")

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
