import sys

import unittest
import torch

#from artist import ARTIST_ROOT
# from ... import ARTIST_ROOT
# from ..data_generator import DataGenerator
# from ..dataset_loader import DataLoader
# from ..datapoint import HeliostatDataPoint, HeliostatDataPointLabel
# from ...scenario.light_source.sun import Sun

from artist import ARTIST_ROOT
from artist.io.data_generator import DataGenerator
from artist.io.dataset_loader import DataLoader
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.scenario.light_source.sun import Sun



class TestDataGeneratorAndLoader(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datapoint = HeliostatDataPoint(
            point_id=1,
            light_directions=torch.tensor([0.0, 0.0, 1.0]),
            desired_aimpoint=torch.tensor([0, -10, 0]),
            label=HeliostatDataPointLabel(),
        )
        self.sun = Sun("Normal", 1, [0, 0], [[4.3681e-06, 0], [0, 4.3681e-06]], device)
        self.dataGenerator = DataGenerator()
        ARTIST_ROOT = '/'.join(__file__.split('/')[:-2])
        print('/'.join(__file__.split('/')[:-2]))

    def test_generate(self):
        expected = {"targetHeliostat": self.datapoint, "lightSource": self.sun}
        filename = f"{ARTIST_ROOT}/scenario_objects/heliostats/test_heliostat.pickle"
        DataGenerator.generate(self, filename, self.datapoint, self.sun)
        self.dataloader = DataLoader()
        data = DataLoader.load_helio_file(filename=filename)
        self.assertEqual(
            data["targetHeliostat"].point_id, expected["targetHeliostat"].point_id
        )
        self.assertEqual(
            data["targetHeliostat"].light_directions[0],
            expected["targetHeliostat"].light_directions[0],
        )
        self.assertEqual(
            data["targetHeliostat"].desired_aimpoint[0],
            expected["targetHeliostat"].desired_aimpoint[0],
        )
        self.assertEqual(
            data["lightSource"].dist_type, expected["lightSource"].dist_type
        )
        self.assertEqual(data["lightSource"].num_rays, expected["lightSource"].num_rays)


if __name__ == "__main__":
    unittest.main()
