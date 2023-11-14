import typing
import torch
from .module import AModule

class AKinematicModule(AModule):
    def __init__(self, position: torch.Tensor):
        super(AKinematicModule, self).__init__()
        self._position = position

    def computeOrientation(
        self, data_point_tensor: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must Be Overridden!")

    def computeReflection(self, data_point_tensor: torch.Tensor):
        normal_vectors, normal_origins = self.computeOrientation(
            data_point_tensor=data_point_tensor
        )
        aim_vectors = (
            2 * (data_point_tensor[:, 2:5] @ normal_vectors) - data_point_tensor[:, 2:5]
        )
        aimpoints = (
            self._position
            + aim_vectors * (data_point_tensor[:, 5:] - self._position).norm()
        )
        return aimpoints

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        return self.computeReflection(data_points)

    def to_dict(self):
        return self.state_dict()