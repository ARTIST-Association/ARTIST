from collections import deque
from typing import Deque

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


class EarlyStopping:
    """
    Implement early stopping.

    Stops optimization when the loss improvement trend over the last few epochs
    falls below a given threshold.

    Attributes
    ----------
    window_size : int
        Number of epochs used to estimate loss trend (default is 10).
    patience : int
        Number of consecutive non-improving windows before stopping (default is 20).
    min_improvement : float
        Minimum required improvement over the window to reset patience (default is 1e-4).
    relative : bool
        Indicates whether improvement is normalized by loss magnitude (default is True).
    eps : float
        Small value for stability (default is 1e-8).
    loss_history : Deque
        Loss values of the past epochs.
    counter : int
        Counter for the epochs.

    Methods
    -------
    step()
        Update stopping state.
    """

    def __init__(
        self,
        window_size: int = 10,
        patience: int = 20,
        min_improvement: float = 1e-4,
        relative: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the early stopping.

        Parameters
        ----------
        window_size : int
            Number of epochs used to estimate loss trend (default is 10).
        patience : int
            Number of consecutive non-improving windows before stopping (default is 20).
        min_improvement : float
            Minimum required improvement over the window to reset patience (default is 1e-4).
        relative : bool
            Indicates whether improvement is normalized by loss magnitude (default is True).
        eps : float
            Small value for stability (default is 1e-8).
        """
        self.window_size = window_size
        self.patience = patience
        self.min_improvement = min_improvement
        self.relative = relative
        self.eps = eps

        self.loss_history: Deque[float] = deque(maxlen=window_size)
        self.counter = 0

    def step(self, loss: float) -> bool:
        """
        Update stopping state.

        Parameters
        ----------
        loss : float
            Current loss value.

        Returns
        -------
        bool
            True if optimization should stop, otherwise False.
        """
        self.loss_history.append(loss)

        if len(self.loss_history) < self.window_size:
            return False

        improvement = self.loss_history[0] - self.loss_history[-1]

        if self.relative:
            improvement /= max(abs(self.loss_history[0]), self.eps)

        if improvement > self.min_improvement:
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
