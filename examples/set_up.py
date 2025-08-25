import pathlib

import h5py
import torch
from mpi4py import MPI

from artist.core import learning_rate_schedulers, loss_functions
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util.environment_setup import get_device


def surface_reconstructor_for_hpo(params: dict[str, float]) -> float:
    """
    Set up a surface reconstructor used in a hyperparameter search.

    Parameters
    ----------
    params : dict[str, float]

    Returns
    -------
    float
        The loss for a specific parameter configuration.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Get device.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    device = get_device(device)

    # Set up ARTIST to run in single device mode.
    ddp_setup = {
        "device": device,
        "is_distributed": False,
        "is_nested": False,
        "rank": 0,
        "world_size": 1,
        "process_subgroup": None,
        "groups_to_ranks_mapping": {0: [...]},
        "heliostat_group_rank": 0,
        "heliostat_group_world_size": 1,
        "ranks_to_groups_mapping": {0: [0], 1: [0]},
    }

    # For parameter combinations with too many rays (over 3000000) directly return a default loss,
    # to avoid running this combination and to avoid causing "out of memory" errors.
    total_number_of_rays = (
        params["number_of_surface_points"]
        * 2
        * 4
        * params["number_of_rays"]
        * params["number_of_training_samples"]
    )
    if total_number_of_rays >= 3000000:
        loss = 987987
        return loss

    number_of_surface_points_per_facet = torch.tensor(
        [params["number_of_surface_points"], params["number_of_surface_points"]],
        device=device,
    )

    number_of_control_points_per_facet = torch.tensor(
        [params["number_of_control_points"], params["number_of_control_points"]],
        device=device,
    )

    # Load the scenario.
    with h5py.File(
        pathlib.Path("examples/data/scenarios/scenario.h5"), "r"
    ) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=number_of_surface_points_per_facet,
            change_number_of_control_points_per_facet=number_of_control_points_per_facet,
            device=device,
        )

    # Set number of rays.
    scenario.light_sources.light_source_list[0].number_of_rays = params[
        "number_of_rays"
    ]

    # Set nurbs degree.
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        heliostat_group.nurbs_degrees = torch.tensor(
            [params["nurbs_degree"], params["nurbs_degree"]], device=device
        )

    # Create a heliostat data mapping for the specified number of training samples.
    heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="/base/path/to/PAINT",
        heliostat_names=["AA39"],
        number_of_measurements=params["number_of_training_samples"],
        image_variant="flux-centered",
    )

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        number_of_surface_points=number_of_surface_points_per_facet,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        flux_loss_weight=1.0,
        ideal_surface_loss_weight=params["ideal_surface_loss_weight"],
        total_variation_loss_points_weight=params["total_variation_loss_points_weight"],
        total_variation_loss_normals_weight=params[
            "total_variation_loss_normals_weight"
        ],
        number_of_neighbors_tv_loss=params["total_variation_loss_number_of_neighbors"],
        sigma_tv_loss=params["total_variation_loss_sigma"],
        early_stopping_threshold=1e-3,
        initial_learning_rate=params["initial_learning_rate"],
        scheduler=getattr(learning_rate_schedulers, params["scheduler"]),
        scheduler_parameters=params,
        tolerance=0.00005,
        max_epoch=3000,
        num_log=1,
        device=device,
    )

    # Reconstruct surfaces.
    loss = surface_reconstructor.reconstruct_surfaces(
        loss_function=getattr(loss_functions, params["loss_function"]), device=device
    )

    return loss[0].item()
