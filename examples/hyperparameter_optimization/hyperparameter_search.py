import logging
import pathlib
import random

import h5py
import torch
from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import (
    parse_arguments,
)

from artist.core import loss_functions
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


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
        config_dictionary.groups_to_ranks_mapping: {0: [0, 1]},
        config_dictionary.heliostat_group_rank: 0,
        config_dictionary.heliostat_group_world_size: 1,
        config_dictionary.ranks_to_groups_mapping: {0: [0], 1:[0]},
    }

    # For parameter combinations with too many rays directly return a default loss,
    # to avoid running such combination as they cause "out of memory" errors.
    total_number_of_rays = (
        params["number_of_surface_points"]
        * 2
        * 4
        * params["number_of_rays"]
        * params["number_of_training_samples"]
    )
    if total_number_of_rays >= 1500000:
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

    # Load a scenario from an .h5 file.
    # The scenario .h5 file should contain a setup with at least one heliostat (with the same name(s)
    # as the heliostat(s) for which reconstruction data is provided). The heliostat(s) in this scenario
    # should be initialized with an ideal surface, do not provide deflectometry data!
    with h5py.File(pathlib.Path("path/to/scenario/scenario.h5"), "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=number_of_surface_points_per_facet,
            change_number_of_control_points_per_facet=number_of_control_points_per_facet,
            device=device,
        )

    # Set number of rays.
    scenario.set_number_of_rays(number_of_rays=int(params["number_of_rays"]))

    # Set nurbs degree.
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        heliostat_group.nurbs_degrees = torch.tensor(
            [params["nurbs_degree"], params["nurbs_degree"]], device=device
        )

    # Create a heliostat data mapping for the specified number of training samples.
    heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="/path/to/data/paint",
        heliostat_names=["AA39"],
        number_of_measurements=int(params["number_of_training_samples"]),
        image_variant="flux-centered",
        randomize=False,
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
        config_dictionary.max_epoch: 1000,
        config_dictionary.num_log: 1,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 2500,
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
        loss_functions.PixelLoss(scenario=scenario)
        if loss_class is loss_functions.PixelLoss
        else loss_functions.KLDivergenceLoss()
    )

    # Reconstruct surfaces.
    loss = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition,
        device=device,
    )

    return loss[torch.isfinite(loss)].sum().item()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    print(rank)

    # Parse command-line arguments.
    config, _ = parse_arguments(comm)
    log_path = "/path/to/logs"

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,
        log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",
        log_to_stdout=False,
        log_rank=True,
        colors=True,
    )

    log = logging.getLogger(__name__)
    rank = comm.Get_rank()
    log.info(rank)

    search_space = {
        "number_of_surface_points": (30, 110),
        "number_of_control_points": (4, 100),
        "number_of_rays": (10, 200),
        "number_of_training_samples": (2, 6),
        "nurbs_degree": (2, 3),
        "scheduler": ("exponential", "cyclic", "reduce_on_plateau"),
        "lr_gamma": (0.85, 0.999),
        "lr_min": (1e-6, 1e-3),
        "lr_max": (1e-3, 1e-1),
        "lr_step_size_up": (100, 2000),
        "lr_reduce_factor": (0.1, 0.7),
        "lr_patience": (5, 50),
        "lr_threshold": (1e-5, 1e-2),
        "lr_cooldown": (0, 20),
        "ideal_surface_loss_weight": (0.00, 1.00),
        "total_variation_loss_points_weight": (0.00, 1.00),
        "total_variation_loss_normals_weight": (0.00, 1.00),
        "total_variation_loss_number_of_neighbors_points": (0, 5000),
        "total_variation_loss_sigma_points": (0, 1),
        "total_variation_loss_number_of_neighbors_normals": (0, 5000),
        "total_variation_loss_sigma_normals": (0, 1),
        "initial_learning_rate": (1e-7, 0.9),
        "loss_class": ("KLDivergenceLoss", "PixelLoss"),
    }

    seed = 7
    rng = random.Random(seed + comm.rank)

    # Set up evolutionary operator.
    num_generations = 500
    pop_size = 2 * comm.size  # Breeding population size
    propagator = get_default_propagator(
        pop_size=pop_size,
        limits=search_space,
        crossover_prob=0.7,
        mutation_prob=0.4,
        random_init_prob=0.1,
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=surface_reconstructor_for_hpo,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=num_generations,  # Number of generations per worker
        checkpoint_path=log_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate(
        logging_interval=1,
        debug=2,  # Logging interval and verbosity level
    )
    propulator.summarize(
        top_n=10,
        debug=2,  # Print top-n best individuals on each island in summary.
    )
