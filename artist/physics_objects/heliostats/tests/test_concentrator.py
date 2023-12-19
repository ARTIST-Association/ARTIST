
import os
import unittest
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

import torch
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import PointCloudFacetModule

from artist.physics_objects.heliostats.surface.tests import surface_defaults
from artist.scenario.light_source.sun import Sun


class TestConcentrator(unittest.TestCase):
    def setUp(self):
        self.position = torch.Tensor([0.0, 5.0, 0.0])
        self.aim_point = torch.Tensor([0.0, -50.0, 0.0])
        self.sun_direction = torch.Tensor([0.0, 0.0, 1.0])
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.surface_config = surface_defaults.load_config_file(cfg_default_surface)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #cov = 0.1e-12
        cov = 4.3681e-06
        self.sun = Sun(
            "Normal", 100, [0, 0], [[cov, 0], [0, cov]], self.device
            #"Normal", 1, [0, 0], [[0.1, 0], [0, 0.1]], self.device
            #"Normal", 1, [0, 0], [[math.sqrt(4.3681e-06), 0], [0, math.sqrt(4.3681e-06)]], self.device
        )
        
        point_cloud_facets = PointCloudFacetModule(self.surface_config, self.aim_point, self.sun_direction, self.device)
        facets = point_cloud_facets.make_facets_list()
        concentrator = ConcentratorModule(facets)
        self.surface_points, self.surface_normals = concentrator.get_surface()

        self.alignment_model = AlignmentModule(position=self.position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.sun_direction,
            desired_aimpoint=self.aim_point,
            label=HeliostatDataPointLabel(),
        )
        print(self.surface_points)
        (
            self.aligned_surface_points,
            self.aligned_surface_normals,
        ) = self.alignment_model.align_surface(
            datapoint=self.datapoint,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
        )
        print(self.aligned_surface_points)


    def test_compute_rays(self):
        torch.manual_seed(7)
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.aim_point
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

        
        #total_bitmap = total_bitmap.T
 
        # ax.scatter(total_bitmap[:, 0], total_bitmap[:, 1], total_bitmap[:, 2], color='red', alpha=0.6)

        new_xlim = (-5, 5)  # New range for x-axis
        new_ylim = (-5, 5)  # New range for y-axis 
 

        
        extent=new_xlim + new_ylim

        plt.imshow(total_bitmap.T, cmap="jet", extent=extent, origin="lower")
        plt.grid(True)
        # new_ticks = [0,64,128,192,256]# Adjust these values based on your scaling
        # new_tick_labels = [0,2,4,6,8, 10]# Adjust these labels based on your desired scale # Set the ticks and labels on both axes 
        # plt.xticks(new_ticks, new_tick_labels) 
        # plt.yticks(new_ticks, new_tick_labels) 
        


        # plt.scatter(128, 128)
        plt.show()

        # torch.save(total_bitmap, 'artist\physics_objects\heliostats\\tests\\bitmaps\\testMap.pt')
        # plt.imshow(total_bitmap.detach().numpy(), cmap="jet")
        # plt.show()

        # expected_path = os.path.join(
        #     "artist", "physics_objects", "heliostats", "tests", "test_bitmaps", "south.txt"
        # )
        # path_to_temp = os.path.join(
        #     "artist", "physics_objects", "heliostats", "tests", "test_bitmaps", "temp.csv"
        # )

        # expected = pd.read_csv(expected_path, delimiter="\t").to_csv(path_to_temp)
        # expected = pd.read_csv(path_to_temp).iloc[1:].iloc[:, 2:]
        # expected = torch.tensor(expected.values)

        # #expected = torch.load(expected_path)
        # plt.imshow(expected, cmap="jet")
        # plt.show()



        # plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(self.surface_points[:, 0], self.surface_points[:, 1], self.surface_points[:, 2], color='blue', alpha=0.6)
        #ax.scatter(self.surface_normals[0, 0], self.surface_normals[0, 1], self.surface_normals[0, 2], color='blue', alpha=0.5)
        ax.scatter(self.aligned_surface_points[:, 0], self.aligned_surface_points[:, 1], self.aligned_surface_points[:, 2], color='red', alpha=0.6)
        #ax.scatter(self.aligned_surface_normals[0, 0], self.aligned_surface_normals[0, 1], self.aligned_surface_normals[0, 2], color='red', alpha=0.5)
        
        # for i, point in enumerate(self.surface_points):
        #     x, y, z = point
        #     u, v, w = self.surface_normals[i]
        #     ax.quiver(x, y, z, u, v, w, length=1, normalize=True, color='blue')

        # for i, point in enumerate(self.aligned_surface_points):
        #     x, y, z = point
        #     u, v, w = self.aligned_surface_normals[i]
        #     ax.quiver(x, y, z, u, v, w, length=1, normalize=True, color='red')

        # # for i, point in enumerate(self.aligned_surface_points):
        # #     x, y, z = point
        # #     u, v, w = ray_directions[i]
        # #     ax.quiver(x, y, z, u, v, w, length=1, normalize=True, color='green')
         
        # for i, point in enumerate(self.aligned_surface_points):
        #     for ray in rays:
        #         x, y, z = point
        #         u, v, w = ray[i]
        #         ax.quiver(x, y, z, u, v, w, length=1, normalize=True, color='orange')


        sun_pos = self.position + (sun_position * 5)
        ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], color='orange', alpha=1, s=300)
        ax.quiver(sun_pos[0], sun_pos[1], sun_pos[2], -sun_position[0], -sun_position[1], -sun_position[2], length=5, normalize=True, color='orange')


        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        plt.show()

if __name__ == "__main__":
    unittest.main()
