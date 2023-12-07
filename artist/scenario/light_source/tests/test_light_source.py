import math
from matplotlib import pyplot as plt
import unittest
import numpy
import pandas as pd
import torch
from artist import ARTIST_ROOT
import numpy as np

from artist.scenario.light_source.sun import Sun
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule


class TestASunModule(unittest.TestCase):
    def setUp(self):
        self.light_direction = torch.tensor([0.0, -1.0, 1.0])
        self.heliostat_position = torch.tensor([0.0, 5.0, 0.0])
        self.receiver_center = torch.tensor([0.0, -10.0, 0.0])

        cov = 1e-12       #4.3681e-06
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sun = Sun(
            "Normal", 300, [0, 0], [[cov, 0], [0, cov]], device
            #"Normal", 1, [0, 0], [[1, 0], [0, 1]], device
        )

        self.surface_normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        )
        self.surface_points = torch.tensor(
            [[-1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [0, 0, 0]]
        )
        self.alignment_model = AlignmentModule(position=self.heliostat_position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.light_direction,
            desired_aimpoint=self.receiver_center,
            label=HeliostatDataPointLabel(),
        )
        (
            self.aligned_surface_points,
            self.aligned_surface_normals,
        ) = self.alignment_model.align_surface(
            datapoint=self.datapoint,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
        )

    def test_compute_rays(self):
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.receiver_center
        receiver_plane_x = 10
        receiver_plane_y = 10
        receiver_resolution_x = 256
        receiver_resolution_y = 256
        sun_position = self.light_direction

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

        total_bitmap = total_bitmap.T
        
        # torch.set_printoptions(threshold=10_000)
        # print(total_bitmap)

        # stral_df = pd.read_csv("artist\physics_objects\heliostats\\tests\\fdm_max_marlene.txt", delimiter="\t", header=1)
        # stral_df.to_csv("artist\physics_objects\heliostats\\tests\\fdm_max_marlene.csv", encoding='utf-8', index=False)

        # stral_df = pd.read_csv("artist\physics_objects\heliostats\\tests\\fdm_max_marlene.csv")
        # # # plt.imshow(stral_df, cmap="jet")
        # # # plt.show()
        
        # stral_df_rotated = numpy.rot90(stral_df, 0)
        # # plt.imshow(stral_df_rotated, cmap="jet")
        # # plt.show()

        # stral_tensor = torch.Tensor(stral_df_rotated.copy())#[:, 1:]
        # stral_tensor_rotated_normalized = self.sun.normalize_bitmap(
        #     stral_tensor,
        #     xi.numel(),
        #     receiver_plane_x,
        #     receiver_plane_y,
        # )

        # tick_positions = np.linspace(0, 150, 11)  # Define 11 tick positions between 0 and 256
        # tick_labels = np.linspace(-1.5, 1.5, 11)     # Define corresponding labels from 0 to 10

        # # Set the ticks and labels for both x and y axes
        # plt.xticks(tick_positions, tick_labels)
        # plt.yticks(tick_positions, tick_labels) 
        # plt.imshow((stral_tensor_rotated_normalized), cmap="jet")
        # plt.grid(True)
        # plt.show()


        fig, ax = plt.subplots(figsize=(6,6))
        tick_positions = np.linspace(0, 256, 11)  # Define 11 tick positions between 0 and 256
        tick_labels = np.linspace(5, 5, 11)     # Define corresponding labels from 0 to 10

        # Set the ticks and labels for both x and y axes
        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels) 
        plt.imshow(total_bitmap.detach().numpy(), aspect="equal", cmap="jet")
        plt.grid(True)
        plt.show()

        # loss = torch.nn.L1Loss()
        # l = loss(total_bitmap, stral_tensor_rotated_normalized)
        # print(l)
        # plt.xticks(tick_positions, tick_labels)
        # plt.yticks(tick_positions, tick_labels) 
        # plt.imshow(total_bitmap - stral_tensor_rotated_normalized)
        # plt.grid(True)
        # plt.show()


        # expected = torch.load(
        #     f"{ARTIST_ROOT}/artist/scenario/light_source/tests/bitmaps/testMap.pt"
        # )

        # torch.testing.assert_close(total_bitmap, expected)


if __name__ == "__main__":
    unittest.main()
