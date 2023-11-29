import unittest
from matplotlib import pyplot as plt

import torch
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule

from artist.physics_objects.heliostats.surface.facets.nurbs_facets import NURBSFacetModule
from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import PointCloudFacetModule
from artist.physics_objects.heliostats.surface.tests import surface_defaults
from artist.scenario.light_source.sun import Sun


class TestNURBS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.position = torch.Tensor([0.0, 0.0, 0.0])
        self.aimpoint = torch.Tensor([0.0, -50.0, 0.0])
        self.sun_directions = torch.Tensor([0.0, 0.0, 1.0])
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.surface_config = surface_defaults.load_config_file(cfg_default_surface)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sun = Sun(
            "Normal", 1, [0, 0], [[4.3681e-06, 0], [0, 4.3681e-06]], self.device
        )

        #nurbs_facet_module = NURBSFacetModule(self.surface_config, self.aimpoint, self.sun_directions, self.device)
        point_cloud_facet = PointCloudFacetModule(self.surface_config, self.aimpoint, self.sun_directions, self.device)
        surface_points, surface_normals = point_cloud_facet.discrete_points_and_normals()
        
        surface_points = point_cloud_facet.facetted_discrete_points
        surface_normals = point_cloud_facet.facetted_normals

        surface_points = torch.cat((surface_points[0], surface_points[1], surface_points[2], surface_points[3]), 0)
        surface_normals = torch.cat((surface_normals[0], surface_normals[1], surface_normals[2], surface_normals[3]) ,0)

        print(surface_points)
        print(surface_normals)

        self.alignmentModel = AlignmentModule(position=self.position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.sun_directions,
            desired_aimpoint=self.aimpoint,
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

    def test_surface(self):
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.aimpoint
        receiver_plane_x = 8.629666667
        receiver_plane_y = 7.0
        receiver_resolution_x = 64
        receiver_resolution_y = 64
        sun_position = self.sun_directions

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