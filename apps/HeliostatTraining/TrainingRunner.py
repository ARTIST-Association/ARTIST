# system dependencies
import sys
import os

import typing
import torch

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)
import HeliostatTrainingLib.AlignmentTrainer as AT

def main(training_dir: typing.Optional[str] = None):
    dtype = torch.float64
    device = torch.device('cpu')
    torch.set_default_dtype(dtype)
    #training_dir = training_dir if training_dir else os.path.abspath(os.path.join(__file__, os.pardir, 'TrainingBase'))
    training_dir = training_dir if training_dir else '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results'
    
    trainer = AT.SIMPLEAlignmentTrainer(dtype=torch.float64)
    training_manager = AT.AlignmentTrainingManager(trainer=trainer, output_dir = training_dir, dtype=dtype, device=device, create_plots=True)
    training_manager.runScheduledTrainings()

if __name__ == '__main__':
    main()