import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from artist.field.nurbs import NURBSSurface


def test_nurbs():
    # control_points = [
    #     [[1.0, 0.0, 0.0, 1.0], [0.7071, 0.7071, 0.0, 0.7071], [0.0, 1.0, 0.0, 1.0], [-0.7071, 0.7071, 0.0, 0.7071], [-1.0, 0.0, 0.0, 1.0], [-0.7071, -0.7071, 0.0, 0.7071], [0.0, -1.0, 0.0, 1.0], [0.7071, -0.7071, 0.0, 0.7071], [1.0, 0.0, 0.0, 1.0]],
    #     [[1.0, 0.0, 1.0, 1.0], [0.7071, 0.7071, 0.7071, 0.7071], [0.0, 1.0, 1.0, 1.0], [-0.7071, 0.7071, 0.7071, 0.7071], [-1.0, 0.0, 1.0, 1.0], [-0.7071, -0.7071, 0.7071, 0.7071], [0.0, -1.0, 1.0, 1.0], [0.7071, -0.7071, 0.7071, 0.7071], [1.0, 0.0, 1.0, 1.0]]
    # ]

    # # Set degrees
    # degree_x = 1
    # degree_y = 2

    # # Set knot vectors
    # knots_x = [0, 0, 1, 1]
    # knots_y = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]

    # nurbs = NURBSSurface(degree_x, degree_y, 40, 40)
    # surface_points, surface_normals = nurbs.calculate_surface_points_and_normals(control_points, knots_x, knots_y)
        
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Define the range of x and y coordinates
    x_range = torch.linspace(-5, 5, 40)
    y_range = torch.linspace(-5, 5, 40)

    # Create a grid of x and y coordinates
    X, Y = torch.meshgrid(x_range, y_range)

    # Define the range of z coordinates (now 2D)
    z_range = torch.linspace(-5, 5, 40)  # Reduce the number of points for z to avoid memory issues

    # Create a grid of z coordinates
    Z1, Z2 = torch.meshgrid(z_range, z_range)

    # Stack Z1 and Z2 to get a 2D Z
    Z = torch.stack((Z1, Z2), dim=-1)

    # Define a function to generate random coefficients for sine and cosine functions
    def generate_random_coefficients():
        return torch.randn(6)  # 6 coefficients for sine and cosine terms


    factor = 0.1
    # Define the random surface function
    def random_surface(x, y, z, coefficients):
        a, b, c, d, e, f = coefficients
        return (factor * a * torch.sin(x) + factor *b * torch.sin(y) + factor*c * torch.sin(z[..., 0]) +
                factor * d * torch.cos(x) + factor *e * torch.cos(y) + factor*f * torch.cos(z[..., 1]))

    # Generate random coefficients for the surface
    surface_coefficients = generate_random_coefficients()

    # Compute the surface using the random coefficients
    surface = random_surface(X, Y, Z, surface_coefficients)
    surface_points = torch.stack((X.flatten(), Y.flatten(), surface.flatten()), dim=-1)
    ones = torch.ones(surface_points.shape[0], 1)
    surface_points = torch.cat((surface_points, ones), dim=1)

    x = surface_points[:, 0]
    y = surface_points[:, 1]
    z = surface_points[:, 2]


    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor="none")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-2, 2])
    #plt.show()


    num_control_points_x = 7
    num_control_points_y = 7

    degree_x = 2
    degree_y = 2

    next_degree_x = degree_x + 1                                                          
    next_degree_y = degree_y + 1 
    
    control_points_shape = (num_control_points_x, num_control_points_y)                       
    control_points = torch.zeros(
        control_points_shape + (3,),                                                  
    )
    origin_offsets_x = torch.linspace(
        -5, 5, num_control_points_x)
    origin_offsets_y = torch.linspace(
        -5, 5, num_control_points_y)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack((
        origin_offsets,
        torch.zeros((len(origin_offsets), 1)),
    ))
    control_points = torch.nn.parameter.Parameter((origin_offsets).reshape(control_points.shape)) 

    knots_x = torch.zeros(num_control_points_x + next_degree_x)                                                                                              
    num_knot_vals = len(knots_x[degree_x:-degree_x])
    knot_vals = torch.linspace(0, 1, num_knot_vals)
    knots_x[:degree_x] = 0
    knots_x[degree_x:-degree_x] = knot_vals
    knots_x[-degree_x:] = 1

    knots_y = torch.zeros(num_control_points_y + next_degree_y)                                                                                        
    num_knot_vals = len(knots_y[degree_y:-degree_y])
    knot_vals = torch.linspace(0, 1, num_knot_vals)
    knots_y[:degree_y] = 0
    knots_y[degree_y:-degree_y] = knot_vals
    knots_y[-degree_y:] = 1

    nurbs = NURBSSurface(degree_x, degree_y, 40, 40)

    optimizer = torch.optim.Adam([control_points], lr=1e-2)

    for epoch in range(150):
        points, normals = nurbs.calculate_surface_points_and_normals(control_points, knots_x, knots_y)

        optimizer.zero_grad()

        #normals = torch.cat((normals, torch.zeros(normals.shape[0], 1)), dim = 1)

        #loss =  points[:, 2] - surface_points[:, 2]
        loss =  points - surface_points
        loss.abs().sum().backward()

        optimizer.step()

        print(loss.abs().sum())

        #loss = (output-control_points_real).sum()
        #print(loss)
    
    #diff = torch.abs(surface_points[:, 2] - points[:, 2])
    
    # mean_loss = loss.abs().sum() / points.shape[0]
    # print(mean_loss)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a meshgrid from coordinates
    X, Y, Z = torch.meshgrid(x, y, z)

    # Plot the 3D tensor
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), cmap='viridis', edgecolor="none")
    #ax.quiver(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), normals[:, 0].detach().numpy(), normals[:, 1].detach().numpy(), normals[:, 2].detach().numpy(), length=0.5, normalize=True)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-2, 2])
    #plt.show()

    # x = diff[:, 0]
    # y = diff[:, 1]
    # z = diff[:, 2]

    # # Create a meshgrid from coordinates
    # X, Y, Z = torch.meshgrid(x, y, z)

    # # Plot the 3D tensor
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), cmap='viridis', edgecolor="none")
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-2, 2])
    
    #print(torch.testing.assert_allclose(points, surface_points))
    plt.show()

