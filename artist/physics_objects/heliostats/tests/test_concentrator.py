import unittest
from matplotlib import pyplot as plt

import torch
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import PointCloudFacetModule

from artist.physics_objects.heliostats.surface.tests import surface_defaults
from artist.scenario.light_source.sun import Sun


class TestConcentrator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.position = torch.Tensor([0.0, 0.0, 0.0])
        self.aim_point =torch.Tensor([0.0, -50.0, 0.0])
        self.sun_direction = torch.Tensor([0.0, 0.0, 1.0])
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.surface_config = surface_defaults.load_config_file(cfg_default_surface)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sun = Sun(
            "Normal", 1, [0, 0], [[4.3681e-06, 0], [0, 4.3681e-06]], self.device
        )
        
        point_cloud_facets = PointCloudFacetModule(self.surface_config, self.aim_point, self.sun_direction, self.device)
        facets = point_cloud_facets.make_facets_list()
        concentrator = ConcentratorModule(facets)
        surface_points, surface_normals = concentrator.get_surface()

        self.alignmentModel = AlignmentModule(position=self.position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.sun_direction,
            desired_aimpoint=self.aim_point,
            label=HeliostatDataPointLabel(),
        )
        (
            self.aligned_surface_points,
            self.aligned_surface_normals,
        ) = self.alignmentModel.align_surface(
            datapoint=self.datapoint,
            surface_points=surface_points,
            surface_normals=surface_normals,
        )

    def test_compute_rays(self):
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.aim_point
        receiver_plane_x = 8.629666667
        receiver_plane_y = 7.0
        receiver_resolution_x = 256
        receiver_resolution_y = 256
        sun_position = self.sun_direction

        ray_directions = self.sun.reflect_rays_(
            -sun_position, self.aligned_surface_normals
        )
        xi, yi = self.sun.sample(1)

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

        plt.imshow(total_bitmap.detach().numpy())
        plt.show()

if __name__ == "__main__":
    unittest.main()
