import matplotlib.pyplot as plt
from matplotlib import cm
import torch as th


def plot_surface_diff(hel_origin, ideal_normal_vecs, target_normal_vectors):
    differences = th.sum(ideal_normal_vecs * target_normal_vectors, dim=-1).detach().cpu()

    x = hel_origin[:,0].detach().cpu()
    y = hel_origin[:,1].detach().cpu()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(0.9999, 1.0001)
    surf = ax.plot_trisurf(x, y, differences, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    exit()

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

def plot_bitmap(bitmap):
    plt.imshow(bitmap.detach().cpu().numpy(), cmap='jet')
    plt.show()
    exit()
