import logging
import pathlib
import random

from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import (
    parse_arguments,
)

from examples.hyperparameter_optimization import set_up

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # Parse command-line arguments.
    config, _ = parse_arguments(comm)
    log_path = "path/to/log/dir"

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,
        log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",
        log_to_stdout=False,
        log_rank=True,
        colors=True,
    )

    search_space = {
        "number_of_surface_points": (30, 100),
        "number_of_control_points": (4, 100),
        "number_of_rays": (10, 200),
        "number_of_training_samples": (2, 16),
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
    num_generations = 100
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
