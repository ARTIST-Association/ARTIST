import torch


def mean_loss_per_heliostat(
    loss_per_sample: torch.Tensor,
    number_of_samples_per_heliostat: int,
) -> torch.Tensor:
    """
    Calculate the mean loss per heliostat from a loss per sample.

    Parameters
    ----------
    loss_per_sample : torch.Tensor
        Loss per sample.
        Tensor of shape [number_of_samples].
    number_of_samples_per_heliostat : int
        Number of samples per heliostat.

    Returns
    -------
    torch.Tensor
        Loss per heliostat.
        Tensor of shape [number_of_heliostats].
    """
    number_of_heliostats = int(
        loss_per_sample.numel() // number_of_samples_per_heliostat
    )
    loss_per_sample = loss_per_sample[
        : number_of_heliostats * number_of_samples_per_heliostat
    ]

    loss_per_heliostat = loss_per_sample.view(
        number_of_heliostats, number_of_samples_per_heliostat
    ).mean(dim=1)

    return loss_per_heliostat
