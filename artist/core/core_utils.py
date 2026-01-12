import torch

from artist.util.environment_setup import get_device


def scale_loss(
    loss: torch.Tensor, reference: torch.Tensor, weight: float
) -> torch.Tensor:
    """
    Scale one loss so that its weighted contribution is a ratio of the reference loss.

    Parameters
    ----------
    loss : torch.Tensor
        The loss to be scaled.
        Tensor of shape [1].
    reference : torch.Tensor
        The reference loss.
        Tensor of shape [1].
    weight : float
        The weight or ratio used for the scaling.

    Returns
    -------
    torch.Tensor
        The scaled loss.
        Tensor of shape [1].
    """
    epsilon = 1e-12
    scale = (reference * weight) / (loss + epsilon)
    scaled_loss = loss * scale

    inf_mask = torch.isinf(loss)
    scaled_loss[inf_mask] = loss[inf_mask]

    return scaled_loss


def reduce_gradients(parameters, process_group=None, mean=True):
    """
    Manually reduce gradients across all ranks.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor]
        Iterable of tensors with .grad attributes.
    process_group : torch.distributed.ProcessGroup | None
        Optional subgroup to reduce over (defaults to the global process group).
    mean : bool
        Whether to divide the reduced gradients by world size (default is True).
    """
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size(group=process_group)
    if world_size == 1:
        return

    with torch.no_grad():
        for param in parameters:
            if param.grad is None:
                continue

            torch.distributed.all_reduce(
                param.grad, op=torch.distributed.ReduceOp.SUM, group=process_group
            )

            if mean:
                param.grad /= world_size


def loss_per_heliostat_distributed(
    local_loss_per_sample, samples_per_heliostat, ddp_setup, device=None
):
    """
    Gather per-sample losses from all ranks to rank 0, and compute per-object loss.

    Parameters
    ----------
    local_loss_per_sample : torch.Tensor
        Tensor of shape [num_local_samples] containing per-sample losses on this rank.
    samples_per_object : torch.Tensor
        Tensor of shape [num_objects] indicating how many samples belong to each object.
    device : torch.device or None
        Device to place the final tensor on. Defaults to local tensor device.

    Returns
    -------
    torch.Tensor or None
        Tensor of shape [num_objects] with per-object losses on rank 0.
        Returns None on other ranks.
    """
    device = get_device(device=device)

    rank = ddp_setup["heliostat_group_rank"]
    world_size = ddp_setup["heliostat_group_world_size"]
    process_subgroup = ddp_setup["process_subgroup"]

    if not torch.distributed.is_initialized() or world_size == 1:
        final_loss_per_heliostat = torch.empty(
            len(samples_per_heliostat), device=device
        )
        start_index = 0
        for i, number_of_samples in enumerate(samples_per_heliostat):
            if number_of_samples > 0:
                heliostat_losses = local_loss_per_sample[
                    start_index : start_index + number_of_samples
                ]
                final_loss_per_heliostat[i] = heliostat_losses.mean()
            else:
                final_loss_per_heliostat[i] = float("nan")
            start_index += number_of_samples
        return final_loss_per_heliostat

    local_number_of_samples = torch.tensor(
        [local_loss_per_sample.numel()], device=device
    )
    max_number_of_samples = local_number_of_samples.clone()
    torch.distributed.all_reduce(
        max_number_of_samples,
        op=torch.distributed.ReduceOp.MAX,
        group=process_subgroup,
    )
    max_number_of_samples = max_number_of_samples.item()

    if local_loss_per_sample.numel() < max_number_of_samples:
        padded = torch.zeros(
            max_number_of_samples, dtype=local_loss_per_sample.dtype, device=device
        )
        padded[: local_loss_per_sample.numel()] = local_loss_per_sample
    else:
        padded = local_loss_per_sample

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded, group=process_subgroup)

    if rank == 0:
        all_losses = []
        for i, size_tensor in enumerate(
            [local_number_of_samples for _ in range(world_size)]
        ):
            size = size_tensor.item()
            all_losses.extend(gathered[i][:size].tolist())

        final_loss_per_heliostat = torch.empty(
            len(samples_per_heliostat), device=device
        )
        start_index = 0
        for i, number_of_samples in enumerate(samples_per_heliostat):
            if number_of_samples > 0:
                heliostat_losses = all_losses[
                    start_index : start_index + number_of_samples
                ]
                final_loss_per_heliostat[i] = torch.tensor(
                    heliostat_losses, device=device
                ).mean()
            else:
                final_loss_per_heliostat[i] = float("nan")
            start_index += number_of_samples

        return final_loss_per_heliostat
    else:
        return None


def mean_loss_per_heliostat(
    loss_per_sample: torch.Tensor,
    nonzero_active_heliostats_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = get_device(device=device)

    indices = torch.repeat_interleave(
        torch.arange(nonzero_active_heliostats_mask.size(0), device=device),
        nonzero_active_heliostats_mask,
    )

    sum_per_heliostat = torch.zeros(len(nonzero_active_heliostats_mask), device=device)
    sum_per_heliostat.scatter_add_(0, indices, loss_per_sample)

    mean_per_heliostat = sum_per_heliostat / nonzero_active_heliostats_mask

    return mean_per_heliostat
