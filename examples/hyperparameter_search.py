import logging
import pathlib
import random

from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import (
    parse_arguments,
)

from artist import ARTIST_ROOT
from examples import set_up

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # Parse command-line arguments.
    config, _ = parse_arguments(comm)

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    scenario_paths = [
        str(scenario_path)
        for scenario_path in (
            pathlib.Path(ARTIST_ROOT) / "examples/data/scenarios"
        ).iterdir()
        if scenario_path.is_file()
        and scenario_path.name.startswith("scenario_")
        and scenario_path.name.endswith("_control_points.h5")
    ]

    search_space = {
        "scenario_path": tuple(scenario_paths),
        "number_of_rays": (10, 200),
        "nurbs_degree": (2, 3),
        "number_of_training_samples": (2, 16),
        "scheduler": ("exponential", "cyclic", "reduce_on_plateau"),
        "lr_gamma": (0.99000, 0.99999),
        "lr_min": (1e-7, 0.9),
        "lr_max": (1e-4, 1.5e-4),
        "lr_step_size_up": (50, 200),
        "lr_reduce_factor": (0.3, 0.5),  # 0.1-0.9
        "lr_patience": (20, 40),  # 10-50
        "lr_threshold": (1e-4, 1e-1),
        "lr_cooldown": (5, 20),
        "number_of_surface_points": (30, 100),  # TODO might be too much memory.
        "ideal_surface_loss_weight": (1e-4, 9e-1),
        "total_variation_loss_points_weight": (1e-4, 9e-1),
        "total_variation_loss_normals_weight": (1e-4, 9e-1),
        "total_variation_loss_number_of_neighbors": (0, 5000),
        "total_variation_loss_sigma": (0, 1),
        "initial_learning_rate": (1e-7, 0.9),
        "loss_function": ("distribution_loss_kl_divergence", "pixel_loss"),
    }

    seed = 2
    rng = random.Random(
        seed + comm.rank
    )  # Separate random number generator for optimization.
    # Set up evolutionary operator.
    num_generations = 5000  # Number of generations
    pop_size = 2 * comm.size  # Breeding population size
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding pool size
        limits=search_space,  # Search-space limits
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=set_up.surface_reconstructor_for_hpo,
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
        top_n=1,
        debug=2,  # Print top-n best individuals on each island in summary.
    )
