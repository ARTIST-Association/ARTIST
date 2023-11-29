import unittest
import torch
import math

from artist.physics_objects.heliostats.alignment.neural_network_rigid_body_fusion import NeuralNetworkRigidBodyFusion
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel


class TestAKinematicModule(unittest.TestCase):
    def setUp(self):
        position = torch.tensor([0, 0, 0])
        self.kinematic_model = NeuralNetworkRigidBodyFusion(position=position)
        self.datapoint1 = HeliostatDataPoint(
            point_id=1,
            light_directions=torch.tensor([0.0, 0.0, 1.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.datapoint2 = HeliostatDataPoint(
            point_id=2,
            light_directions=torch.tensor([1.0, 0.0, 0.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.datapoint3 = HeliostatDataPoint(
            point_id=3,
            light_directions=torch.tensor([0.0, -1.0, 0.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.datapoint4 = HeliostatDataPoint(
            point_id=4,
            light_directions=torch.tensor([-1.0, 0.0, 0.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.datapoint5 = HeliostatDataPoint(
            point_id=5,
            light_directions=torch.tensor([0.0, -1.0, 1.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.datapoint6 = HeliostatDataPoint(
            point_id=6,
            light_directions=torch.tensor([0.0, 0.0, 1.0]),
            desired_aimpoint=torch.tensor([0, -9, 0]),
            label=HeliostatDataPointLabel(),
        )
        position2 = torch.tensor([0.0, 1.0, 0.0])
        self.kinematic_model2 = NeuralNetworkRigidBodyFusion(position=position2)

        self.datapoint7 = HeliostatDataPoint(
            point_id=7,
            light_directions=torch.tensor([0.0, -10.0, 0.0]),
            desired_aimpoint=torch.tensor([0, -50, 0]),
            label=HeliostatDataPointLabel(),
        )

    def test_compute_orientation_from_aimpoint1(self):
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint1
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    def test_compute_orientation_from_aimpoint2(self):
        expected = torch.tensor(
            [
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint2
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    def test_compute_orientation_from_aimpoint3(self):
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint3
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    def test_compute_orientation_from_aimpoint4(self):
        expected = torch.tensor(
            [
                [1 / math.sqrt(2), -1 / math.sqrt(2), 0.0, 0.0],
                [-1 / math.sqrt(2), -1 / math.sqrt(2), 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint4
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    def test_compute_orientation_from_aimpoint5(self):
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -math.cos(math.pi / 8), -math.sin(math.pi / 8), 0.0],
                [0.0, math.sin(math.pi / 8), -math.cos(math.pi / 8), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint5
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    def test_compute_orientation_from_aimpoint6(self):
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -math.cos(math.pi / 8), -math.sin(math.pi / 8), 1.0],
                [0.0, math.sin(math.pi / 8), -math.cos(math.pi / 8), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        orientation_matrix = self.kinematic_model2.compute_orientation_from_aimpoint(
            self.datapoint6
        )
        torch.testing.assert_close(orientation_matrix[0], expected)

    
    def test_compute_orientation_from_aimpoint7(self):
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        orientation_matrix = self.kinematic_model.compute_orientation_from_aimpoint(
            self.datapoint7
        )
        torch.testing.assert_close(orientation_matrix[0], expected)


if __name__ == "__main__":
    unittest.main()
