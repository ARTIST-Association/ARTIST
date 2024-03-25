
import torch
from artist.physics_objects.actuator import AActuatorModule


class LinearActuator(AActuatorModule):
    """
    This class implements the behavior of a linear actuator.

    Attributes
    ----------
    joint_number : int
        Descriptor (number) of the joint.
    clockwise : bool
        Turning direction of the joint.
    increment : torch.Tensor
        The increment of an actuator.
    initial_stroke_length : torch.Tensor
        The initial_stroke_length.
    actuator_offset : torch.Tensor
        The actuator offset.
    joint_radius : toorch.Tensor
        The joint radius.
    phi_0 : torch.Tensor
        An initial angle.

    Methods
    -------
    steps_to_phi()
        Calculate phi (angle) from steps.
    motor_steps_to_angles()
        Calculate the angles given the motor steps.
    angles_to_motor_steps()
        Calculate the motor steps given the angles.
    forward()
        Perform the forward kinematic.

    See Also
    --------
    :class:`AActuatorModule` : The parent class.
    """
    def __init__(self, 
                 joint_number: int, 
                 clockwise: bool, 
                 increment: torch.Tensor, 
                 initial_stroke_length: torch.Tensor,
                 actuator_offset: torch.Tensor,
                 joint_radius: torch.Tensor,
                 phi_0: torch.Tensor) -> None:
        """
        Parameters
        ----------
        joint_number : int
            Descriptor (number) of the joint.
        clockwise : bool
            Turning direction of the joint.
        increment : torch.Tensor
            The increment of an actuator.
        initial_stroke_length : torch.Tensor
            The initial_stroke_length.
        actuator_offset : torch.Tensor
            The actuator offset.
        joint_radius : toorch.Tensor
            The joint radius.
        phi_0 : torch.Tensor
            An initial angle.
        """
        super().__init__(joint_number, clockwise, increment, initial_stroke_length, actuator_offset, joint_radius, phi_0)
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.actuator_offset = actuator_offset
        self.joint_radius = joint_radius
        self.phi_0 = phi_0


    def steps_to_phi(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Calculate phi (angle) from steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator position.
        
        Returns
        -------
        torch.Tensor
            The calculated angle.
        """
        stroke_length = (
            actuator_pos / self.increment
            + self.initial_stroke_length
        )
        calc_step_1 = (
            self.actuator_offset ** 2
            + self.joint_radius ** 2
            - stroke_length**2
        )
        calc_step_2 = 2.0 * self.actuator_offset * self.joint_radius
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle
    
    def motor_steps_to_angles(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Calculate the angles given the motor steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator (motor) position.
        
        Returns
        -------
        torch.Tensor
            The angles corresponding to the motor steps.
        """
        phi = self.steps_to_phi(actuator_pos=actuator_pos)
        phi_0 = self.steps_to_phi(actuator_pos=torch.zeros(actuator_pos.shape))
        delta_phi = phi_0 - phi

        angles = self.phi_0 + delta_phi if self.clockwise else self.phi_0 - delta_phi
        return angles
    
    def angles_to_motor_steps(self, angles : torch.Tensor) -> torch.Tensor:
        """
        Calculate the motor steps given the angles.

        Parameters
        ----------
        angles : torch.Tensor
            The angles.
        
        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        delta_phi = angles - self.phi_0 if self.clockwise else self.phi_0 - angles
        
        phi_0 = self.steps_to_phi(actuator_pos=torch.zeros(angles.shape[0]))
        phi = phi_0 - delta_phi
        
        calc_step_3 = torch.cos(phi)
        calc_step_2 = 2.0 * self.actuator_offset * self.joint_radius
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt( self.actuator_offset ** 2  + self.joint_radius ** 2 - calc_step_1)
        actuator_steps = (stroke_length - self.initial_stroke_length) * self.increment
        return actuator_steps

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator (motor) position.
        
        Returns
        -------
        torch.Tensor
            The angles.
        """
        return self.motor_steps_to_angles(actuator_pos=actuator_pos)