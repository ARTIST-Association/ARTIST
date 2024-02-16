import torch
from artist.physics_objects.heliostats.alignment.kinematic.actuators.actuator import AActuatorModule


class IdealActuator(AActuatorModule):    
    
    def motor_steps_to_angles(self, motor_steps_joint_1: torch.Tensor, motor_steps_joint_2: torch.Tensor):
        
        return angles
    
    def angles_to_motor_steps(self, angles):
        
        return motor_steps