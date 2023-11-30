from matplotlib import pyplot as plt
import unittest
import torch
from artist import ARTIST_ROOT

from artist.scenario.light_source.sun import Sun
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule


class TestASunModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.light_direction = torch.tensor([0.0, 0.0, 1.0])
        self.heliostat_position = torch.tensor([0.0, 1.0, 0.0])
        self.receiver_center = torch.tensor([0.0, -50.0, 0.0])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sun = Sun(
            "Normal", 1, [0, 0], [[0.0000000001, 0], [0, 0.0000000001]], device
        )
        surface_normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        )
        surface_points = torch.tensor(
            [[-1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [0, 0, 0]]
        )
        self.alignmentModel = AlignmentModule(position=self.heliostat_position)
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=self.light_direction,
            desired_aimpoint=self.receiver_center,
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
        print(self.aligned_surface_points)
        print(self.aligned_surface_normals)

    def test_compute_rays(self):
        receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
        receiver_center = self.receiver_center
        receiver_plane_x = 8.629666667
        receiver_plane_y = 7.0
        receiver_resolution_x = 64
        receiver_resolution_y = 64
        sun_position = self.light_direction

        # sun_position = sun_position.float()
        # self.aligned_surface_normals = self.aligned_surface_normals.float()

        ray_directions = self.sun.reflect_rays_(
            -sun_position, self.aligned_surface_normals
        )
        print(ray_directions)
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

        total_bitmap = total_bitmap.T
        
        torch.set_printoptions(threshold=10_000)
        #print(total_bitmap)

        plt.imshow(total_bitmap.detach().numpy())
        plt.show()

        expected = torch.load(
            f"{ARTIST_ROOT}/artist/scenario/light_source/tests/bitmaps/testMap.pt"
        )

        # torch.testing.assert_close(total_bitmap, expected)


if __name__ == "__main__":
    unittest.main()
