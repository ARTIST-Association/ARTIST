import datetime
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HeliostatDataPointLabel(object):
    """
    Heliostat data point label.

    Attributes
    ----------
    bitmap : Optional[torch.Tensor]
        The bitmap.
    orientation : Optional[torch.Tensor]
        The orientation.
    measured_aimpoint : Optional[torch.Tensor]
        The measured aimpoint.
    """

    bitmap: Optional[torch.Tensor] = None
    orientation: Optional[torch.Tensor] = None
    measured_aimpoint: Optional[torch.Tensor] = None


@dataclass
class HeliostatDataPoint(object):
    """
    Heliostat data point sample, characterizing environmental features.

    Attributes
    ----------
    point_id : int
        The point ID.
    light_directions : torch.Tensor
        The light directions.
    desired_aimpoint : torch.Tensor
        The desired aimpoint.
    label : HeliostatDataPointLabel
        The data point's label.
    heliostat_pos : Optional[torch.Tensor]
        The heliostat position.
    motor_pos : Optional[torch.Tensor]
        The motor position.
    timestamp : Optional[datetime.datetime]
        The time stamp.
    """

    point_id: int
    light_directions: torch.Tensor
    desired_aimpoint: torch.Tensor
    label: HeliostatDataPointLabel
    heliostat_pos: Optional[torch.Tensor] = None
    motor_pos: Optional[torch.Tensor] = None
    timestamp: Optional[datetime.datetime] = None
