import torch

from artist.util import index_mapping
from artist.util.environment_setup import get_device


def decompose_rotations(
    initial_vector: torch.Tensor,
    target_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ENU components of the axis-angle rotation vector that rotates initial vectors toward a target vector.

    This function does **not** perform an Euler-angle decomposition. Instead, it computes:
    1) the rotation axis via cross product, and
    2) the rotation magnitude via arccos of the dot product,
    then returns the Cartesian components of the rotation vector
    (`theta * axis`) in east, north, and up coordinates.

    Parameters
    ----------
    initial_vector : torch.Tensor
        Initial vectors in homogeneous coordinates.
        Shape is ``[number_of_heliostats, 4]``.
        Only the first three components (ENU) are used.
    target_vector : torch.Tensor
        Target vector in homogeneous coordinates.
        Shape is ``[4]``.
        Only the first three components (ENU) are used.

    Returns
    -------
    torch.Tensor
        East component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    torch.Tensor
        North component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    torch.Tensor
        Up component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    """
    # Normalize the input vectors.
    initial_vector = torch.nn.functional.normalize(
        initial_vector[:, : index_mapping.slice_fourth_dimension]
    )
    target_vector = torch.nn.functional.normalize(
        target_vector[: index_mapping.slice_fourth_dimension],
        dim=index_mapping.unbatched_tensor_values,
    ).unsqueeze(index_mapping.unbatched_tensor_values)

    # Compute the cross product (rotation axis).
    r = torch.linalg.cross(initial_vector, target_vector)

    # Normalize the rotation axis.
    r_normalized = torch.nn.functional.normalize(r)

    # Compute the angle between the vectors.
    theta = torch.arccos(torch.clamp(initial_vector @ target_vector.T, -1.0, 1.0))

    # Decompose the angle along each axis.
    theta_components = theta * r_normalized

    return theta_components[:, 0], theta_components[:, 1], theta_components[:, 2]


def rotation_angle_and_axis(
    from_orientation: torch.Tensor,
    to_orientation: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotation axis and angle between two orientations.

    Parameters
    ----------
    from_orientation : torch.Tensor
        The original orientation.
        Shape is ``[4]``.
    to_orientation : torch.Tensor
        The rotated orientation.
        Shape is ``[4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The rotation axis.
        Shape is ``[3]``.
    torch.Tensor
        The angle of the rotation as a scalar tensor.
    """
    device = get_device(device=device)
    from_orientation = from_orientation[:3] / torch.norm(from_orientation[:3])
    to_orientation = to_orientation[:3] / torch.norm(to_orientation[:3])
    dot = torch.clamp(torch.dot(from_orientation, to_orientation), -1.0, 1.0)
    angle = torch.acos(dot)
    axis = torch.linalg.cross(from_orientation, to_orientation)
    axis_norm = torch.norm(axis)
    # Parallel vectors.
    epsilon = 1e-6
    if axis_norm < epsilon and dot > 0:
        return torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(
            0.0, device=device
        )
    # Inverse vectors.
    if axis_norm < epsilon and dot < 0:
        if abs(from_orientation[index_mapping.e]) < abs(
            from_orientation[index_mapping.n]
        ):
            orthogonal = torch.tensor([1.0, 0.0, 0.0], device=device)
        else:
            orthogonal = torch.tensor([0.0, 1.0, 0.0], device=device)
        axis = torch.linalg.cross(from_orientation, orthogonal)
        axis = axis / torch.norm(axis)
        return axis, torch.tensor(torch.pi, device=device)
    axis = axis / axis_norm
    return axis, angle
