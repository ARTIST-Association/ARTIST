import pathlib
from artist import ARTIST_ROOT
from artist.util.alignment_optimizer import AlignmentOptimizer
from scripts import utils
import torch


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # The distributed environment is setup and destroyed using a Generator object.
    environment_generator = utils.setup_distributed_environment(device=device)
    
    is_distributed, rank, world_size = next(environment_generator)

    if device.type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    scenario_path = pathlib.Path(ARTIST_ROOT) / "tests/data/test_scenario_paint.h5"
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
    )

    alignment_optimizer = AlignmentOptimizer(
        scenario_path=scenario_path,
        calibration_properties_path=calibration_properties_path,
    )

    optimized_parameters, _ = alignment_optimizer.optimize_kinematic_parameters_with_raytracing(
        device=device, world_size=world_size, rank=rank, batch_size=100, is_distributed=is_distributed
    )

