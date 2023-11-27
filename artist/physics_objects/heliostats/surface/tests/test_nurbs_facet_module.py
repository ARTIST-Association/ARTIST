import unittest

import torch

from artist.physics_objects.heliostats.surface.facets.nurbs_facets import NURBSFacetModule
from artist.physics_objects.heliostats.surface.tests import surface_defaults


class TestNURBS(unittest.TestCase):
    def setUp(self):
        self.aimpoint = torch.Tensor([0.0, 10.0, 0.0])
        self.sun_directions = torch.Tensor([0.0, 0.0, 1.0])
        
        cfg_default_surface = surface_defaults.get_cfg_defaults()
        self.surface_config = surface_defaults.load_config_file(cfg_default_surface)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_nurbs_facet_module(self):
        nurbsFacetModule = NURBSFacetModule(self.surface_config, self.aimpoint, self.sun_directions, self.device)


if __name__ == "__main__":
    unittest.main()