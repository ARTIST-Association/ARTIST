# system dependencies
import sys
import os

import typing
import torch

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)
import HeliostatTrainingLib.AlignmentTrainer as AT
import HeliostatKinematicLib.AlignmentDisturbanceModel as ADM

def main(training_dir: typing.Optional[str] = None):
    dtype = torch.float32
    device = torch.device('cpu')

    torch.set_default_dtype(dtype)
    # training_dir = training_dir if training_dir else os.path.abspath(os.path.join(__file__, os.pardir, 'TrainingBase'))
    training_dir = training_dir if training_dir else '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results'

    # rigid body disturbances
    disturbance_list = [
        'position_azim',
        'position_elev',
        'position_rad',

        'joint_2_cosys_pivot_azim',
        'joint_2_cosys_pivot_elev',
        'joint_2_cosys_pivot_rad',

        'concentrator_cosys_pivot_azim',
        'concentrator_cosys_pivot_elev',
        'concentrator_cosys_pivot_rad',

        'joint_1_east_tilt',
        'joint_1_north_tilt',
        'joint_1_up_tilt',

        'joint_2_east_tilt',
        'joint_2_north_tilt',
        'joint_2_up_tilt',

        'concentrator_east_tilt',
        'concentrator_north_tilt',
        'concentrator_up_tilt',

        'actuator_1_increment',
        'actuator_2_increment',
    ]

    pre_alignment_model = ADM.RigidBodyAlignmentDisturbanceModel(disturbance_list = disturbance_list, dtype=dtype, device=device)

    trainer = AT.SIMPLEAlignmentTrainer(dtype=torch.float64)
    training_manager = AT.AlignmentTrainingManager(trainer=trainer, output_dir = training_dir, dtype=dtype, device=device, create_plots=True)
    training_manager.runScheduledTrainings(pre_alignment_model=pre_alignment_model)

if __name__ == '__main__':
    main()