from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from artist.util import constants
from artist.util.env import get_device


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
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(parameters[constants.gamma])
    )


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
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=float(parameters[constants.lr_min]),
        max_lr=float(parameters[constants.lr_max]),
        step_size_up=int(parameters[constants.step_size_up]),
    )


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
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(parameters[constants.reduce_factor]),
        patience=int(parameters[constants.patience]),
        threshold=float(parameters[constants.threshold]),
        cooldown=int(parameters[constants.cooldown]),
        min_lr=float(parameters[constants.lr_min]),
    )


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


@dataclass
class TrainTestSplit:
    """
    Container holding the train/test split for heliostat reconstruction data.

    Attributes
    ----------
    flux_measured_train : torch.Tensor
        Measured flux distributions for the training set.
        Shape is ``[number_of_train_samples_total, height, width]``.
    focal_spots_measured_train : torch.Tensor
        Measured focal spot coordinates for the training set.
        Shape is ``[number_of_train_samples_total, 4]``.
    incident_ray_directions_train : torch.Tensor
        Incident ray directions for the training set.
        Shape is ``[number_of_train_samples_total, 4]``.
    motor_positions_train : torch.Tensor
        Motor positions for the training set.
        Shape is ``[number_of_train_samples_total, 2]``.
    target_area_indices_train : torch.Tensor
        Target area indices for the training set.
        Shape is ``[number_of_train_samples_total]``.
    flux_measured_test : torch.Tensor
        Measured flux distributions for the test set.
        Shape is ``[number_of_test_samples_total, height, width]``.
    focal_spots_measured_test : torch.Tensor
        Measured focal spot coordinates for the test set.
        Shape is ``[number_of_test_samples_total, 4]``.
    incident_ray_directions_test : torch.Tensor
        Incident ray directions for the test set.
        Shape is ``[number_of_test_samples_total, 4]``.
    motor_positions_test : torch.Tensor
        Motor positions for the test set.
        Shape is ``[number_of_test_samples_total, 2]``.
    target_area_indices_test : torch.Tensor
        Target area indices for the test set.
        Shape is ``[number_of_test_samples_total]``.
    active_heliostats_mask_train : torch.Tensor
        Mask for active training samples available per heliostat after splitting.
        Shape is ``[number_of_heliostats]``.
    active_heliostats_mask_test : torch.Tensor
        Mask for active test samples available per heliostat after splitting.
        Shape is ``[number_of_heliostats]``.
    train_indices : torch.Tensor
        Indices of training samples from the original dataset.
        Shape is ``[number_of_train_samples_total]``.
    test_indices : torch.Tensor
        Indices of test samples from the original dataset.
        Shape is ``[number_of_test_samples_total]``.
    number_of_train_samples : int
        Number of training samples per heliostat.
    number_of_test_samples : int
        Number of test samples per heliostat.
    number_of_samples_per_heliostat : int
        Total number of samples available per heliostat before splitting.
    """

    flux_measured_train: torch.Tensor
    focal_spots_measured_train: torch.Tensor
    incident_ray_directions_train: torch.Tensor
    motor_positions_train: torch.Tensor
    target_area_indices_train: torch.Tensor

    flux_measured_test: torch.Tensor
    focal_spots_measured_test: torch.Tensor
    incident_ray_directions_test: torch.Tensor
    motor_positions_test: torch.Tensor
    target_area_indices_test: torch.Tensor

    active_heliostats_mask_train: torch.Tensor
    active_heliostats_mask_test: torch.Tensor

    train_indices: torch.Tensor
    test_indices: torch.Tensor

    number_of_train_samples: int
    number_of_test_samples: int
    number_of_samples_per_heliostat: int


def train_test_split(
    active_heliostats_mask: torch.Tensor,
    flux_measured: torch.Tensor,
    focal_spots_measured: torch.Tensor,
    incident_ray_directions: torch.Tensor,
    motor_positions: torch.Tensor,
    target_area_indices: torch.Tensor,
    test_fraction: float = 0.25,
    device: torch.device | None = None,
) -> TrainTestSplit:
    """
    Split heliostat reconstruction data into training and test subsets.

    The split is performed independently for each heliostat while preserving
    the original ordering of samples. Training samples are taken from the
    beginning of each heliostat block, while test samples are taken from the
    end of each heliostat block.

    Parameters
    ----------
    active_heliostats_mask : torch.Tensor
        Mask for active samples available per heliostat.
        Shape is ``[number_of_heliostats]``.
    flux_measured : torch.Tensor
        Measured flux distributions for all heliostats.
        Shape is ``[total_number_of_samples, height, width]``.
    focal_spots_measured : torch.Tensor
        Measured focal spot coordinates for all samples.
        Shape is ``[total_number_of_samples, 4]``.
    incident_ray_directions : torch.Tensor
        Incident ray directions for all samples.
        Shape is ``[total_number_of_samples, 4]``.
    motor_positions : torch.Tensor
        Motor positions for all samples.
        Shape is ``[total_number_of_samples, 2]``.
    target_area_indices : torch.Tensor
        Integer target area indices for all samples.
        Shape is ``[total_number_of_samples]``.
    test_fraction : float
        Fraction of samples per heliostat assigned to the test set (default is 0.35).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    TrainTestSplit
        Dataclass containing: train/test tensors, train/test indices, updated active heliostat masks.
    """
    device = get_device(device=device)

    total_samples = int(active_heliostats_mask.sum().item())
    number_of_heliostats = int((active_heliostats_mask > 0).sum().item())
    number_of_samples_per_heliostat = int(total_samples / number_of_heliostats)
    number_of_test_samples = max(
        1, int(number_of_samples_per_heliostat * test_fraction)
    )
    number_of_train_samples = number_of_samples_per_heliostat - number_of_test_samples
    starts = (
        torch.arange(number_of_heliostats, device="cpu")
        * number_of_samples_per_heliostat
    )
    offsets = torch.arange(
        number_of_train_samples, number_of_samples_per_heliostat, device="cpu"
    )
    train_indices = (
        torch.arange(0, total_samples, number_of_samples_per_heliostat, device="cpu")[
            :, None
        ]
        + torch.arange(number_of_train_samples, device=device)
    ).reshape(-1)
    test_indices = (starts[:, None] + offsets).reshape(-1)

    active_heliostats_mask_train = torch.clamp(
        active_heliostats_mask - number_of_test_samples, min=0
    )
    active_heliostats_mask_test = torch.clamp(
        active_heliostats_mask - number_of_train_samples, min=0
    )

    return TrainTestSplit(
        flux_measured_train=flux_measured[train_indices],
        focal_spots_measured_train=focal_spots_measured[train_indices],
        incident_ray_directions_train=incident_ray_directions[train_indices],
        motor_positions_train=motor_positions[train_indices],
        target_area_indices_train=target_area_indices[train_indices],
        flux_measured_test=flux_measured[test_indices],
        focal_spots_measured_test=focal_spots_measured[test_indices],
        incident_ray_directions_test=incident_ray_directions[test_indices],
        motor_positions_test=motor_positions[test_indices],
        target_area_indices_test=target_area_indices[test_indices],
        active_heliostats_mask_train=active_heliostats_mask_train,
        active_heliostats_mask_test=active_heliostats_mask_test,
        train_indices=train_indices,
        test_indices=test_indices,
        number_of_train_samples=number_of_train_samples,
        number_of_test_samples=number_of_test_samples,
        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
    )
