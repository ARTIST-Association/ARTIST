
import os
import unittest
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

import torch
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
import matplotlib.pyplot as plt
import numpy as np


import os
import unittest
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

import torch
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.surface.facets.nurbs_facets import NURBSFacetsModule

from artist.physics_objects.heliostats.surface.tests import surface_defaults, nurbs_defaults
from artist.scenario.light_source.sun import Sun


from matplotlib.widgets import Button, Slider


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def setUp():
    position = torch.Tensor([0.0, 0.0, 0.0])
    receiver_center = torch.Tensor([0.0, -50, 0.0])
    sun_direction = torch.Tensor([0.0, -1.0, 0.0])
    
    cfg_default_surface = surface_defaults.get_cfg_defaults()
    surface_config = surface_defaults.load_config_file(cfg_default_surface)

    cfg_default_nurbs = nurbs_defaults.get_cfg_defaults()
    nurbs_config = nurbs_defaults.load_config_file(cfg_default_nurbs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #cov = 0.1e-12
    cov = 4.3681e-06
    sun = Sun(
        "Normal", 100, [0, 0], [[cov, 0], [0, cov]], device
        #"Normal", 1, [0, 0], [[0.1, 0], [0, 0.1]], device
        #"Normal", 1, [0, 0], [[math.sqrt(4.3681e-06), 0], [0, math.sqrt(4.3681e-06)]], device
    )
    
    nurbs_facets = NURBSFacetsModule(
            surface_config, nurbs_config, device, receiver_center=receiver_center
    )

    facets = nurbs_facets.make_facets_list()
    concentrator = ConcentratorModule(facets)
    surface_points, surface_normals = concentrator.get_surface()

    alignment_model = AlignmentModule(position=position)
    datapoint = HeliostatDataPoint(
        point_id=1,
        light_directions=sun_direction,
        desired_aimpoint=receiver_center,
        label=HeliostatDataPointLabel(),
    )
    # print(surface_points)
    (
        aligned_surface_points,
        aligned_surface_normals,
    ) = alignment_model.align_surface(
        datapoint=datapoint,
        surface_points=surface_points,
        surface_normals=surface_normals,
    )
    # print(aligned_surface_points)
    return sun_direction, sun, aligned_surface_normals, aligned_surface_points

def test_compute_rays(receiver, sun_direction, sun, aligned_surface_normals, aligned_surface_points):
    torch.manual_seed(7)
    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
    receiver_center = receiver
    receiver_plane_x = 10
    receiver_plane_y = 10 
    receiver_resolution_x = 256
    receiver_resolution_y = 256
    sun_position = sun_direction

    ray_directions = sun.reflect_rays_(
        -sun_position, aligned_surface_normals
    )
    
    xi, yi = sun.sample(len(ray_directions))

    rays = sun.compute_rays(
        receiver_plane_normal,
        receiver_center,
        ray_directions,
        aligned_surface_points,
        xi,
        yi,
    )

    intersections = sun.line_plane_intersections(
        receiver_plane_normal, receiver_center, rays, aligned_surface_points
    )

    dx_ints = intersections[:, :, 0] + receiver_plane_x / 2 - receiver_center[0]

    dy_ints = intersections[:, :, 2] + receiver_plane_y / 2 - receiver_center[2]

    indices = (
            (-1 <= dx_ints)
            & (dx_ints < receiver_plane_x + 1)
            & (-1 <= dy_ints)
            & (dy_ints < receiver_plane_y + 1)
    )

    total_bitmap = sun.sample_bitmap(
        dx_ints,
        dy_ints,
        indices,
        receiver_plane_x,
        receiver_plane_y,
        receiver_resolution_x,
        receiver_resolution_y,
    )
    
    total_bitmap = sun.normalize_bitmap(
        total_bitmap,
        xi.numel(),
        receiver_plane_x,
        receiver_plane_y,
    )
    # plt.imshow(total_bitmap.T, cmap="jet", origin="lower")
    # #plt.grid(True)
    # plt.show()

    return total_bitmap

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
sun_direction, sun, aligned_surface_normals, aligned_surface_points = setUp()
image = test_compute_rays(torch.Tensor([0, -50, 0]), sun_direction, sun, aligned_surface_normals, aligned_surface_points)
img = plt.imshow(image.T, cmap="jet", origin="lower")

ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
slider = Slider(ax_slider, 'receiver_center', -300, 0, valinit=-50)

def update(val):
   receiver = torch.Tensor([0.0, slider.val, 0.0])
   sun_direction, sun, aligned_surface_normals, aligned_surface_points = setUp()
   im = test_compute_rays(receiver, sun_direction, sun, aligned_surface_normals, aligned_surface_points).T
   img.set_data(im)
   fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()