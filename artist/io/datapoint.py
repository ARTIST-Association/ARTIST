from dataclasses import dataclass
import datetime
from typing import Optional
import torch

@dataclass
class HeliostatDataPoint(object):
    """
    Represents a single state of a heliostat, encapsulating all relevant information needed for simulation,
    analysis, or control of heliostat fields in solar power plants in a single data point object.

    Attributes:
        point_id (int): A unique identifier for the heliostat data point. This could be used for tracking, referencing,
            or indexing purposes in datasets or simulations.
        light_directions (torch.Tensor): A tensor representing the directions of light incident on the heliostat. This
            could include the sun's position in the sky relative to the heliostat's position and orientation or multiple arrays from a projector.
            dim: (N,3)
        desired_aimpoint (torch.Tensor): A tensor indicating the desired aim point for the heliostat's reflected
            sunlight. This represents the ideal focal spots center position on the target area where the heliostat is
            intended to direct the sunlight.
        label (HeliostatDataPointLabel): An instance of `HeliostatDataPointLabel` containing label information such as
            bitmap, orientation, and measured aim point, useful for training models or analyzing performance.
        heliostat_pos (Optional[torch.Tensor]): A tensor representing the position of the heliostat. 
            Default is None, indicating that the position may not be provided.
        motor_pos (Optional[torch.Tensor]): A tensor indicating the positions of the motors controlling the heliostat's
            orientation. This can provide insight into the heliostat's current orientation or status. 
            Default is None.
        timestamp (Optional[datetime.datetime]): A datetime object representing the time at which the data point was
            recorded.
            Default is None.
    """

@dataclass
class HeliostatDataPointLabel(object):
    """
    A data class representing a label for a heliostat data point.

    Attributes:
        bitmap (Optional[torch.Tensor]): A tensor representing the bitmap (focal spot) image of the heliostat.
            Default is None, indicating that the bitmap may not be provided.
        orientation (Optional[torch.Tensor]): A tensor representing the orientation of the heliostat given by an ENU 4x4 Camera transformation matrix.
            Default is None.
        measured_aimpoint (Optional[torch.Tensor]): A tensor indicating the measured aim point of the heliostat's
            reflection. Default is None.

    This class is typically used to encapsulate various label information possible for a single heliostat in a solar field.
    """
    bitmap: Optional[torch.Tensor] = None
    orientation: Optional[torch.Tensor] = None
    measured_aimpoint: Optional[torch.Tensor] = None

