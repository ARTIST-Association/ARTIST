import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def exponential(
    optimizer: Optimizer,
    parameters: dict[str, float],
) -> LRScheduler:
    """
    Create an exponential learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        The optimzer.
    parameters : dict[str, float]
        The scheduler paramters.

    Returns
    -------
    LRScheduler
        An exponential learning rate scheduler.
    """
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=parameters["lr_gamma"]
    )

    return scheduler


def cyclic(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, float],
) -> LRScheduler:
    """
    Create a cyclic learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        The optimzer.
    parameters : dict[str, float]
        The scheduler paramters.

    Returns
    -------
    LRScheduler
        A cyclic learning rate scheduler.
    """
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=parameters["lr_min"],
        max_lr=parameters["lr_max"],
        step_size_up=parameters["lr_step_size_up"],
    )

    return scheduler


def reduce_on_plateau(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, float],
) -> LRScheduler:
    """
    Create learning rate scheduler that reduces on plateaus.

    Parameters
    ----------
    optimizer : Optimizer
        The optimzer.
    parameters : dict[str, float]
        The scheduler paramters.

    Returns
    -------
    LRScheduler
        A learning rate scheduler that reduces on plateaus.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=parameters["lr_reduce_factor"],
        patience=parameters["lr_patience"],
        threshold=parameters["lr_threshold"],
        cooldown=parameters["lr_cooldown"],
        min_lr=parameters["lr_min"],
    )

    return scheduler


class NoOpScheduler:
    """A no-op learning rate scheduler that does nothing, can be used as a default."""
    
    def __init__(self, **kwargs):
        pass

    def step(self):
        """No-op step function."""
        pass
