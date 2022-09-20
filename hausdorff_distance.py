from typing import cast, List, Tuple, Union

import torch
import torch as th


def contour_images(
        images: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
) -> torch.Tensor:
    contoured = th.zeros_like(images)

    for contour_val in contour_vals:
        contoured[
            (images >= contour_val - contour_val_radius)
            & (images < contour_val + contour_val_radius)
        ] = 1

    return contoured


def segment_images(
        images: torch.Tensor,
        split_vals: List[float],
) -> torch.Tensor:
    split_vals.sort()
    prev_split_val = 0.0
    segmented = th.empty_like(images)

    for (level, split_val) in enumerate(split_vals):
        segmented[(images >= prev_split_val) & (images < split_val)] = level
        prev_split_val = split_val

    segmented[(images >= prev_split_val) & (images <= 1.0)] = level + 1
    return segmented


def isolines_to_sets(contoured_images: torch.Tensor) -> List[torch.Tensor]:
    if contoured_images.ndim < 3:
        contoured_images = contoured_images.unsqueeze(0)

    sets_flat = th.nonzero(contoured_images).to(contoured_images.dtype)
    sets = []
    # TODO here we split the indices depending on the images in the
    #      batch that they belong to. would be good to find a fast
    #      function for this.
    for i in range(len(contoured_images)):
        point_set = sets_flat[sets_flat[:, 0] == i]
        point_set = point_set[:, 1:]
        sets.append(point_set)
    return sets


def images_to_sets(
        images: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
) -> List[torch.Tensor]:
    contoured_images = contour_images(images, contour_vals, contour_val_radius)
    sets = isolines_to_sets(contoured_images)
    return sets


# Adapted from
# https://github.com/gogoymh/Pytorch-Hausdorff-Distance/blob/e496e6ec1799951e9db87465d39ac4b00199a4d6/hausdorff_distance.py.
def set_hausdorff_distance(
        pred_sets: List[torch.Tensor],
        target_sets: List[torch.Tensor],
        norm_p: float = 2.0,
) -> torch.Tensor:
    assert len(pred_sets) > 0
    assert len(pred_sets) == len(target_sets)
    device = pred_sets[0].device
    dtype = pred_sets[0].dtype

    max_dist_ys = th.empty((len(pred_sets), 1), device=device, dtype=dtype)
    max_dist_xs = th.empty((len(pred_sets), 1), device=device, dtype=dtype)
    for (i, (pred_set, target_set)) in enumerate(zip(pred_sets, target_sets)):
        if len(pred_set) == 0 or len(target_set) == 0:
            print(
                'Warning: Hausdorff sets cannot be empty; '
                'assigning Hausdorff distance as infinity...'
            )
            max_dist_y = th.tensor(float('inf'))
            max_dist_x = th.tensor(float('inf'))
        else:
            distance_matrix = th.cdist(
                pred_set.unsqueeze(0),
                target_set.unsqueeze(0),
                p=norm_p,
            )

            max_dist_y = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
            max_dist_x = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
        max_dist_ys[i] = max_dist_y
        max_dist_xs[i] = max_dist_x

    values = th.cat((max_dist_ys, max_dist_xs), dim=1)
    return values.max(1)[0]


def isoline_hausdorff_distance(
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        norm_p: float = 2.0,
) -> torch.Tensor:
    pred_sets = isolines_to_sets(pred_images)
    target_sets = isolines_to_sets(target_images)
    return set_hausdorff_distance(pred_sets, target_sets, norm_p)


def image_hausdorff_distance(
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
        norm_p: float = 2.0,
) -> torch.Tensor:
    pred_images = contour_images(pred_images, contour_vals, contour_val_radius)
    target_images = contour_images(
        target_images, contour_vals, contour_val_radius)
    return isoline_hausdorff_distance(pred_images, target_images, norm_p)


def max_hausdorff_distance(
        image_size: Union[th.Size, Tuple[int, int]],
        device: th.device,
        dtype: th.dtype,
        norm_p: float = 2.0,
) -> torch.Tensor:
    num_pixels = th.prod(th.tensor(image_size))
    first = th.zeros(cast(int, num_pixels), device=device, dtype=dtype)
    first[0] = 1
    last = th.zeros(cast(int, num_pixels), device=device, dtype=dtype)
    last[-1] = 1

    d_max = isoline_hausdorff_distance(
        first.reshape((1,) + image_size),
        last.reshape((1,) + image_size),
        norm_p,
    )
    return d_max


def generalized_mean(
        vals: torch.Tensor,
        dim: int = -1,
        mean_p: float = -1.0,
        epsilon: float = 1e-6
) -> torch.Tensor:
    vals = th.where(
        vals == 0,
        epsilon,
        vals,
    )
    return vals.pow(mean_p).mean(dim).pow(1 / mean_p)


def weighted_hausdorff_distance(
        pred_images: torch.Tensor,
        target_sets: List[torch.Tensor],
        *,
        epsilon: float = 1e-6,
        norm_p: float = 2.0,
        mean_p: float = -1.0,
) -> torch.Tensor:
    assert (
        pred_images.shape[0] > 0
        and pred_images.shape[1] > 0
        and pred_images.shape[2] > 0
    )
    assert len(pred_images) == len(target_sets)

    device = pred_images.device
    dtype = pred_images.dtype

    # Constant over the loop.
    d_max = max_hausdorff_distance(pred_images[0].shape, device, dtype, norm_p)
    pixel_indices = th.cartesian_prod(
        th.arange(pred_images.shape[1], device=device, dtype=dtype),
        th.arange(pred_images.shape[2], device=device, dtype=dtype),
    )

    weighted_dists = th.empty(
        (len(pred_images), 1),
        device=device,
        dtype=dtype,
    )
    for (i, (pred_image, target_set)) in enumerate(zip(
            pred_images,
            target_sets,
    )):
        pred_image_max = pred_image.max()
        if pred_image_max == 0 or len(target_set) == 0:
            print(
                'Warning: weighted Hausdorff sets cannot be empty; '
                'assigning weighted Hausdorff distance as infinity...'
            )
            weighted_dist = th.tensor(float('inf'))
        else:
            pred_image = (pred_image / pred_image_max).reshape(-1, 1)
            pixel_value_sum = pred_image.sum()

            distance_matrix = th.cdist(
                pixel_indices.unsqueeze(0),
                target_set.unsqueeze(0),
                p=norm_p,
            ).squeeze(0)
            min_dist_ys = distance_matrix.min(1)[0]

            point_estimate_loss = (
                (pred_image.squeeze(-1) * min_dist_ys).sum()
                / (pixel_value_sum + epsilon)
            )

            contributions = (
                pred_image * distance_matrix
                + (1 - pred_image) * d_max
            )
            contribution_loss = generalized_mean(
                contributions,
                dim=0,
                mean_p=mean_p,
                epsilon=epsilon,
            ).mean()

            weighted_dist = point_estimate_loss + contribution_loss
        weighted_dists[i] = weighted_dist

    return weighted_dists
