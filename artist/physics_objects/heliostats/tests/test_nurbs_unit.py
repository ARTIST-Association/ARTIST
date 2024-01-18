
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


class TestConcentrator(unittest.TestCase):
    def setUp(self):
        self.position = torch.Tensor([0.0, 5.0, 0.0])
        self.receiver_center = torch.Tensor([0.0, -50, 0.0])
        self.sun_direction = torch.Tensor([0.0, -1.0, 0.0])
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        surface_config = surface_defaults.load_config_file(cfg_default_surface)

        cfg_default_nurbs = nurbs_defaults.get_cfg_defaults()
        nurbs_config = nurbs_defaults.load_config_file(cfg_default_nurbs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #cov = 0.1e-12
        cov = 4.3681e-06
        self.sun = Sun(
            "Normal", 100, [0, 0], [[cov, 0], [0, cov]], self.device
            #"Normal", 1, [0, 0], [[0.1, 0], [0, 0.1]], self.device
            #"Normal", 1, [0, 0], [[math.sqrt(4.3681e-06), 0], [0, math.sqrt(4.3681e-06)]], self.device
        )
        
        nurbs_facets = NURBSFacetsModule(
                surface_config, nurbs_config, self.device, receiver_center=self.receiver_center
        )

        facets = nurbs_facets.make_facets_list()
        concentrator = ConcentratorModule(facets)
        self.surface_points, self.surface_normals = concentrator.get_surface()

        self.alignment_model = AlignmentModule(position=self.position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.sun_direction,
            desired_aimpoint=self.receiver_center,
            label=HeliostatDataPointLabel(),
        )
        # print(self.surface_points)
        (
            self.aligned_surface_points,
            self.aligned_surface_normals,
        ) = self.alignment_model.align_surface(
            datapoint=self.datapoint,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
        )
        # print(self.aligned_surface_points)

    def test_compute_rays(self):
        torch.manual_seed(7)
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.receiver_center
        receiver_plane_x = 10
        receiver_plane_y =10 
        receiver_resolution_x = 256
        receiver_resolution_y = 256
        sun_position = self.sun_direction

        ray_directions = self.sun.reflect_rays_(
            -sun_position, self.aligned_surface_normals
        )
        
        xi, yi = self.sun.sample(len(ray_directions))

        rays = self.sun.compute_rays(
            receiver_plane_normal,
            receiver_center,
            ray_directions,
            self.aligned_surface_points,
            xi,
            yi,
        )

        intersections = self.sun.line_plane_intersections(
            receiver_plane_normal, receiver_center, rays, self.aligned_surface_points
        )

        dx_ints = intersections[:, :, 0] + receiver_plane_x / 2 - receiver_center[0]

        dy_ints = intersections[:, :, 2] + receiver_plane_y / 2 - receiver_center[2]

        indices = (
                (-1 <= dx_ints)
                & (dx_ints < receiver_plane_x + 1)
                & (-1 <= dy_ints)
                & (dy_ints < receiver_plane_y + 1)
        )

        total_bitmap = self.sun.sample_bitmap(
            dx_ints,
            dy_ints,
            indices,
            receiver_plane_x,
            receiver_plane_y,
            receiver_resolution_x,
            receiver_resolution_y,
        )
        
        total_bitmap = self.sun.normalize_bitmap(
            total_bitmap,
            xi.numel(),
            receiver_plane_x,
            receiver_plane_y,
        )
        plt.imshow(total_bitmap.T, cmap="jet", origin="lower")
        # plt.grid(True)
        plt.show()


if __name__ == "__main__":
    unittest.main()
