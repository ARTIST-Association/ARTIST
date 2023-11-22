import unittest

import sys

import torch
sys.path.append( '.' )

from artist.physics_objects.heliostats.surface.nurbs.nurbs_surface import NURBSSurface, MultiNURBSSurface

import nurbs_defaults
import surface_defaults

class TestNURBS(unittest.TestCase):
    def setUp(self):
        cfg_default_nurbs = nurbs_defaults.get_cfg_defaults()
        self.nurbs_config = nurbs_defaults.load_config_file(cfg_default_nurbs)
        cfg_default_nurbs.freeze()
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.heliostat_config = surface_defaults.load_config_file(cfg_default_surface)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        position = torch.Tensor([0.0, 1.0, 0.0])
        self.nurbs_surface = NURBSSurface(self.heliostat_config, self.nurbs_config, device, position)
        self.multi_nurbs_surface = MultiNURBSSurface(self.heliostat_config, self.nurbs_config, device)

    def test_calc_normals_and_surface(self):
        surface_points, surface_normals = self.nurbs_surface.calc_normals_and_surface()
        print(surface_points, surface_normals)

        #print(self.multi_nurbs_surface._create_facets(self.heliostat_config, self.nurbs_config))

if __name__ == "__main__":
    unittest.main()
