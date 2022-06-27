from typing import List

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
