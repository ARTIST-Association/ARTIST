import pathlib

import h5py
import torch
from mpi4py import MPI

from artist.core import loss_functions
from artist.core.loss_functions import KLDivergenceLoss, PixelLoss
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
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
        config_dictionary.device: device,
        config_dictionary.is_distributed: False,
        config_dictionary.is_nested: False,
        config_dictionary.rank: 0,
        config_dictionary.world_size: 1,
        config_dictionary.process_subgroup: None,
        config_dictionary.groups_to_ranks_mapping: {0: [...]},
        config_dictionary.heliostat_group_rank: 0,
        config_dictionary.heliostat_group_world_size: 1,
        config_dictionary.ranks_to_groups_mapping: {0: [0], 1: [0]},
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
    with h5py.File(pathlib.Path("path/to/scenario/scenario.h5"), "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=number_of_surface_points_per_facet,
            change_number_of_control_points_per_facet=number_of_control_points_per_facet,
            device=device,
        )

    # Set number of rays.
    scenario.light_sources.light_source_list[0].number_of_rays = int(
        params["number_of_rays"]
    )

    # Set nurbs degree.
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        heliostat_group.nurbs_degrees = torch.tensor(
            [params["nurbs_degree"], params["nurbs_degree"]], device=device
        )

    # Create a heliostat data mapping for the specified number of training samples.
    heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="path/to/paint/data/dir",
        heliostat_names=["AA39"],
        number_of_measurements=int(params["number_of_training_samples"]),
        image_variant="flux-centered",
    )

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    scheduler = params["scheduler"]
    scheduler_parameters = {
        config_dictionary.gamma: params["lr_gamma"],
        config_dictionary.min: params["lr_min"],
        config_dictionary.max: params["lr_max"],
        config_dictionary.step_size_up: params["lr_step_size_up"],
        config_dictionary.reduce_factor: params["lr_reduce_factor"],
        config_dictionary.patience: params["lr_patience"],
        config_dictionary.threshold: params["lr_threshold"],
        config_dictionary.cooldown: params["lr_cooldown"],
    }

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=params["ideal_surface_loss_weight"], reduction_dimensions=(1, 2, 3, 4)
    )
    total_variation_regularizer_points = TotalVariationRegularizer(
        weight=params["total_variation_loss_points_weight"],
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=int(
            params["total_variation_loss_number_of_neighbors_points"]
        ),
        sigma=params["total_variation_loss_sigma_points"],
    )
    total_variation_regularizer_normals = TotalVariationRegularizer(
        weight=params["total_variation_loss_normals_weight"],
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_normals,
        number_of_neighbors=int(
            params["total_variation_loss_number_of_neighbors_normals"]
        ),
        sigma=params["total_variation_loss_sigma_normals"],
    )

    regularizers = [
        ideal_surface_regularizer,
        total_variation_regularizer_points,
        total_variation_regularizer_normals,
    ]

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: params["initial_learning_rate"],
        config_dictionary.tolerance: 0.00005,
        config_dictionary.max_epoch: 3000,
        config_dictionary.num_log: 1,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 20,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        number_of_surface_points=number_of_surface_points_per_facet,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    # Define loss.
    loss_class = getattr(loss_functions, str(params["loss_class"]))
    loss_definition = (
        PixelLoss(scenario=scenario) if loss_class is PixelLoss else KLDivergenceLoss()
    )

    # Reconstruct surfaces.
    loss = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition,
        device=device,
    )

    return loss[0].item()
