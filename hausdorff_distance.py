from typing import List

import torch
import torch as th


def contour_images(
        image: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
) -> torch.Tensor:
    contoured = th.zeros_like(image)

    for contour_val in contour_vals:
        contoured[
            (image >= contour_val - contour_val_radius)
            & (image < contour_val + contour_val_radius)
        ] = 1

    return contoured


def segment_images(
        image: torch.Tensor,
        split_vals: List[float],
) -> torch.Tensor:
    split_vals.sort()
    prev_split_val = 0.0
    segmented = th.empty_like(image)

    for (level, split_val) in enumerate(split_vals):
        segmented[(image >= prev_split_val) & (image < split_val)] = level
        prev_split_val = split_val

    segmented[(image >= prev_split_val) & (image <= 1.0)] = level + 1
    return segmented


# Adapted from
# https://github.com/gogoymh/Pytorch-Hausdorff-Distance/blob/e496e6ec1799951e9db87465d39ac4b00199a4d6/hausdorff_distance.py.
def hausdorff_distance_contoured(
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        norm_p: float = 2.0,
) -> torch.Tensor:
    distance_matrix = torch.cdist(pred_images, target_images, p=norm_p)

    values1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    values2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    values = torch.cat((values1, values2), dim=1)
    return values.max(1)[0]


def hausdorff_distance(
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        contour_vals: List[float],
        contour_val_radius: float,
        norm_p: float = 2.0,
) -> torch.Tensor:
    pred_images = contour_images(pred_images, contour_vals, contour_val_radius)
    target_images = contour_images(
        target_images, contour_vals, contour_val_radius)
    return hausdorff_distance_contoured(pred_images, target_images, norm_p)
