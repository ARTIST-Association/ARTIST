from dataclasses import dataclass
import datetime
from typing import Optional
import torch


@dataclass
class HeliostatDataPointLabel(object):
    bitmap: Optional[torch.Tensor] = None
    orientation: Optional[torch.Tensor] = None
    measured_aimpoint: Optional[torch.Tensor] = None


@dataclass
class HeliostatDataPoint(object):
    point_id: int
    light_directions: torch.Tensor
    desired_aimpoint: torch.Tensor
    label: HeliostatDataPointLabel
    heliostat_pos: Optional[torch.Tensor] = None
    motor_pos: Optional[torch.Tensor] = None
    timestamp: Optional[datetime.datetime] = None
