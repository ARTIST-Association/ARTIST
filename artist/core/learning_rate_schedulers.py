import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from artist.util import config_dictionary


def exponential(
    optimizer: Optimizer,
    parameters: dict[str, float],
) -> LRScheduler:
    """
    Create an exponential learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer.
    parameters : dict[str, float]
        The scheduler parameters.

    Returns
    -------
    LRScheduler
        An exponential learning rate scheduler.
    """
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(parameters[config_dictionary.gamma])
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
        The optimizer.
    parameters : dict[str, float]
        The scheduler parameters.

    Returns
    -------
    LRScheduler
        A cyclic learning rate scheduler.
    """
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=float(parameters[config_dictionary.min]),
        max_lr=float(parameters[config_dictionary.max]),
        step_size_up=parameters[config_dictionary.step_size_up],
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
        The optimizer.
    parameters : dict[str, float]
        The scheduler parameters.

    Returns
    -------
    LRScheduler
        A learning rate scheduler that reduces on plateaus.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(parameters[config_dictionary.reduce_factor]),
        patience=parameters[config_dictionary.patience],
        threshold=float(parameters[config_dictionary.threshold]),
        cooldown=parameters[config_dictionary.cooldown],
        min_lr=float(parameters[config_dictionary.min]),
    )

    return scheduler
