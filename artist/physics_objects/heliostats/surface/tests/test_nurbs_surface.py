import unittest
# 
import sys
from matplotlib import pyplot as plt

import torch

from artist.scenario.light_source.sun import Sun
sys.path.append( '.' )


from artist.physics_objects.heliostats.surface.nurbs.multi_nurbs_surface import MultiNURBSSurface, NURBSSurface
import nurbs_defaults
import surface_defaults

class TestNURBS(unittest.TestCase):
    def setUp(self):
        cfg_default_nurbs = nurbs_defaults.get_cfg_defaults()
        self.nurbs_config = nurbs_defaults.load_config_file(cfg_default_nurbs)
        cfg_default_nurbs.freeze()
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.heliostat_config = surface_defaults.load_config_file(cfg_default_surface)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        position = torch.Tensor([0.0, 1.0, 0.0])
        self.aim_point =torch.Tensor([0.0, -10.0, 0.0])
        self.sun_direction = torch.Tensor([0.0, 0.0, 1.0])
        self.nurbs_surface = NURBSSurface(self.heliostat_config, self.nurbs_config, self.device, position, self.aim_point)
        
        self.multi_nurbs_surface = MultiNURBSSurface(self.heliostat_config, self.nurbs_config, self.device, position, receiver_center=self.aim_point)

    def test_nurbs_surface(self):
        surface_points, surface_normals = self.nurbs_surface._calc_normals_and_surface()
        print(surface_points, surface_normals)

    def test_multi_nurbs_surface(self):
        surface_points, surface_normals = self.multi_nurbs_surface._calc_normals_and_surface()
        print(surface_points, surface_normals)


if __name__ == "__main__":
    unittest.main()
