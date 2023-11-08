import unittest
import torch
import math
import AKinematicModule
import ADatapoint

class TestAKinematicModule(unittest.TestCase):
    def setUp(self):
        position = torch.tensor([0, 0, 0])
        self.kinematicModel = AKinematicModule.KinematicModule(position=position)
        self.datapoint1 = ADatapoint.HeliostatDataPoint(point_id=1,
                                                      light_directions=torch.tensor([0.0, 0.0, 1.0]),
                                                      desired_aimpoint=torch.tensor([0, -10, 0]),
                                                      label=ADatapoint.HeliostatDataPointLabel())
        self.datapoint2 = ADatapoint.HeliostatDataPoint(point_id=2,
                                                      light_directions=torch.tensor([1.0, 0.0, 0.0]),
                                                      desired_aimpoint=torch.tensor([0, -10, 0]),
                                                      label=ADatapoint.HeliostatDataPointLabel())
        self.datapoint3 = ADatapoint.HeliostatDataPoint(point_id=3,
                                                      light_directions=torch.tensor([0.0, -1.0, 0.0]),
                                                      desired_aimpoint=torch.tensor([0, -10, 0]),
                                                      label=ADatapoint.HeliostatDataPointLabel())
        self.datapoint4 = ADatapoint.HeliostatDataPoint(point_id=4,
                                                      light_directions=torch.tensor([-1.0, 0.0, 0.0]),
                                                      desired_aimpoint=torch.tensor([0, -10, 0]),
                                                      label=ADatapoint.HeliostatDataPointLabel())
        self.datapoint5 = ADatapoint.HeliostatDataPoint(point_id=5,
                                                      light_directions=torch.tensor([0.0, -1.0, 1.0]),
                                                      desired_aimpoint=torch.tensor([0, -10, 0]),
                                                      label=ADatapoint.HeliostatDataPointLabel())
        
    def test_computeOrientationFromAimpoint1(self):
        expected = torch.tensor([[1, 0,                0,              0],
                                 [0, -1/math.sqrt(2), -1/math.sqrt(2), 0],
                                 [0, 1/math.sqrt(2),  -1/math.sqrt(2), 0],
                                 [0, 0,                0,              1]])
        orientationMatrix = self.kinematicModel.computeOrientationFromAimpoint(self.datapoint1)
        print(expected)
        print(orientationMatrix)
        torch.testing.assert_close(orientationMatrix[0], expected)
    
    def test_computeOrientationFromAimpoint2(self):
        expected = torch.tensor([[1/math.sqrt(2), 1/math.sqrt(2),  0,  0],
                                 [1/math.sqrt(2), -1/math.sqrt(2), 0,  0],
                                 [0,              0,               -1, 0],
                                 [0,              0,               0,  1]])
        orientationMatrix = self.kinematicModel.computeOrientationFromAimpoint(self.datapoint2)
        torch.testing.assert_close(orientationMatrix[0], expected)

    def test_computeOrientationFromAimpoint3(self):
        expected = torch.tensor([[1.0, 0.0,  0.0,  0.0],
                                 [0.0, -1.0, 0.0,  0.0],
                                 [0.0, 0.0,  -1.0, 0.0],
                                 [0.0, 0.0,  0.0,  1.0]])
        orientationMatrix = self.kinematicModel.computeOrientationFromAimpoint(self.datapoint3)
        torch.testing.assert_close(orientationMatrix[0], expected)
    
    def test_computeOrientationFromAimpoint4(self):
        expected = torch.tensor([[1/math.sqrt(2),  -1/math.sqrt(2), 0.0,  0.0],
                                 [-1/math.sqrt(2), -1/math.sqrt(2), 0.0,  0.0],
                                 [0.0,             0.0,             -1.0, 0.0],
                                 [0.0,             0.0,             0.0,  1.0]])
        orientationMatrix = self.kinematicModel.computeOrientationFromAimpoint(self.datapoint4)
        torch.testing.assert_close(orientationMatrix[0], expected)

    def test_computeOrientationFromAimpoint5(self):
        expected = torch.tensor([[1.0, 0.0,                  0.0,                  0.0],
                                 [0.0, -math.cos(math.pi/8), -math.sin(math.pi/8), 0.0],
                                 [0.0, math.sin(math.pi/8),  -math.cos(math.pi/8), 0.0],
                                 [0.0, 0.0,                  0.0,                  1.0]])
        orientationMatrix = self.kinematicModel.computeOrientationFromAimpoint(self.datapoint5)
        torch.testing.assert_close(orientationMatrix[0], expected)



if __name__ == '__main__':
    unittest.main()