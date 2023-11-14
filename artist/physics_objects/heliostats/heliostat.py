from artist.physics_objects.heliostats.alignment.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from ..module import AModule


class HeliostatModule(AModule):
    def __init__(self, concentrator: ConcentratorModule, alignment: AlignmentModule):
        self.concentrator = concentrator
        self.alignment = alignment

    def get_aligned_surface(self, datapoint):
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            datapoint, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
