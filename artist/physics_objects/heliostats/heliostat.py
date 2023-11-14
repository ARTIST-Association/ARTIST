import ConcentratorModule
import AlignmentModule
from module import AModule

class HeliostatModule(AModule):
    def __init__(
        self,
        concentrator: ConcentratorModule,
        alignment: AlignmentModule
    ):
        self.concentrator = concentrator
        self.alignment = alignment

    def get_aligned_surface(self, Datapoint):
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(Datapoint, surface_points, surface_normals)
        return aligned_surface_points, aligned_surface_normals