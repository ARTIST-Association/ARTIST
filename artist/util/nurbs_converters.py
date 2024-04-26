from typing import Tuple
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as f
from artist.field.nurbs import NURBSSurface

def deflectometry_to_nurbs(surface_points: torch.Tensor, surface_normals: torch.Tensor, width:torch.Tensor, height: torch.Tensor )-> NURBSSurface:
    
    evaluation_points = surface_points.clone()
    evaluation_points[:,2] = 0

    evaluation_points_x = (evaluation_points[:, 0]-min(evaluation_points[:, 0]) + 1e-5)/max((evaluation_points[:, 0]-min(evaluation_points[:, 0])) + 2e-5)
    evaluation_points_y = (evaluation_points[:, 1]-min(evaluation_points[:, 1]) + 1e-5)/max((evaluation_points[:, 1]-min(evaluation_points[:, 1])) + 2e-5)

    num_control_points_x = 5
    num_control_points_y = 5

    degree_x = 2
    degree_y = 2
    
    control_points_shape = (num_control_points_x, num_control_points_y)                       
    control_points = torch.zeros(
        control_points_shape + (3,),                                                  
    )
    origin_offsets_x = torch.linspace(
        -width/2, width/2, num_control_points_x)
    origin_offsets_y = torch.linspace(
        -height/2, height/2, num_control_points_y)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack((
        origin_offsets,
        torch.zeros((len(origin_offsets), 1)),
    ))

    control_points = torch.nn.parameter.Parameter((origin_offsets).reshape(control_points.shape))
    
    nurbs = NURBSSurface(degree_x, degree_y, evaluation_points_x, evaluation_points_y, control_points)

    optimizer = torch.optim.Adam([control_points], lr=5e-3)

    for epoch in range(1):
        points, normals = nurbs.calculate_surface_points_and_normals()

        optimizer.zero_grad()

        loss = points - surface_points 
        loss.abs().mean().backward()

        optimizer.step()

        print(loss.abs().mean())

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot the 3D tensor
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x.detach().numpy(), y.detach().numpy(), c=z.detach().numpy())
    #ax.quiver(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), normals[:, 0].detach().numpy(), normals[:, 1].detach().numpy(), normals[:, 2].detach().numpy(), length=0.5, normalize=True)
    
    plt.show()

    x = surface_points[:, 0]
    y = surface_points[:, 1]
    z = surface_points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x.detach().numpy(), y.detach().numpy(), z.detach().numpy())
    #.quiver(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), surface_normals[:, 0].detach().numpy(), surface_normals[:, 1].detach().numpy(), surface_normals[:, 2].detach().numpy(), length=1000, normalize=True)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-2, 2])
    plt.show()
    close_ = torch.isclose(points, surface_points)
    print(close_)

    return nurbs
