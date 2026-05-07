import torch

from artist.util import indices
from artist.util.env import get_device


def rotate_distortions(
    e: torch.Tensor,
    u: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotate the distortions for the light source.

    Rotate around the up and then the east axis in this very order in a right-handed east-north-up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.,
    counter-clockwise. Points need to be multiplied as column vectors from the right-hand side with the
    resulting rotation matrix. Note that the order is fixed due to the non-commutative property of matrix-matrix
    multiplication.

    Parameters
    ----------
    e : torch.Tensor
        East rotation angles in radians.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points]``.
    u : torch.Tensor
        Up rotation angles in radians.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the shapes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Batched 4×4 rotation matrices, one per distortion sample.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points, 4, 4]``.
    """
    device = get_device(device=device)

    if e.shape != u.shape:
        raise ValueError(
            "The two tensors containing angles for the east and up rotation must have the same shape."
        )

    cos_e = torch.cos(e)
    sin_e = torch.sin(e)
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    ones = torch.ones_like(e, device=device)

    matrix = torch.zeros(
        e.shape[indices.heliostat_dimension],
        e.shape[indices.facet_dimension],
        e.shape[indices.points_dimension],
        4,
        4,
        device=device,
    )

    matrix[:, :, :, indices.e, indices.e] = cos_u
    matrix[:, :, :, indices.e, indices.n] = -sin_u
    matrix[:, :, :, indices.n, indices.e] = cos_e * sin_u
    matrix[:, :, :, indices.n, indices.n] = cos_e * cos_u
    matrix[:, :, :, indices.n, indices.u] = -sin_e
    matrix[:, :, :, indices.u, indices.e] = sin_e * sin_u
    matrix[:, :, :, indices.u, indices.n] = sin_e * cos_u
    matrix[:, :, :, indices.u, indices.u] = cos_e
    matrix[
        :,
        :,
        :,
        indices.transform_homogeneous,
        indices.transform_homogeneous,
    ] = ones

    return matrix


def rotate_e(
    e: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotate around the east axis.

    Rotate around the east axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors
    from the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    e : torch.Tensor
        East rotation angles in radians.
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 east-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_e = torch.cos(e)
    sin_e = torch.sin(e)
    ones = torch.ones_like(e, device=device)

    matrix = torch.zeros(e.shape[indices.heliostat_dimension], 4, 4, device=device)

    matrix[:, indices.e, indices.e] = ones
    matrix[:, indices.n, indices.n] = cos_e
    matrix[:, indices.n, indices.u] = -sin_e
    matrix[:, indices.u, indices.n] = sin_e
    matrix[:, indices.u, indices.u] = cos_e
    matrix[:, indices.transform_homogeneous, indices.transform_homogeneous] = ones

    return matrix


def rotate_n(n: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """
    Rotate around the north axis.

    Rotate around the north axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors
    from the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    n : torch.Tensor
        North rotation angles in radians.
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 north-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    ones = torch.ones_like(n, device=device)

    matrix = torch.zeros(n.shape[indices.heliostat_dimension], 4, 4, device=device)

    matrix[:, indices.e, indices.e] = cos_n
    matrix[:, indices.e, indices.u] = -sin_n
    matrix[:, indices.n, indices.n] = ones
    matrix[:, indices.u, indices.e] = sin_n
    matrix[:, indices.u, indices.u] = cos_n
    matrix[:, indices.transform_homogeneous, indices.transform_homogeneous] = ones

    return matrix


def rotate_u(u: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """
    Rotate around the up axis.

    Rotate around the up axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation in
    the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors from
    the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    u : torch.Tensor
        Up rotation angles in radians.
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 up-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    ones = torch.ones_like(u, device=device)

    matrix = torch.zeros(u.shape[indices.heliostat_dimension], 4, 4, device=device)

    matrix[:, indices.e, indices.e] = cos_u
    matrix[:, indices.e, indices.n] = -sin_u
    matrix[:, indices.n, indices.e] = sin_u
    matrix[:, indices.n, indices.n] = cos_u
    matrix[:, indices.u, indices.u] = ones
    matrix[:, indices.transform_homogeneous, indices.transform_homogeneous] = ones

    return matrix


def translate_enu(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Translate in all directions.

    Translate a given point in the east, north, and up direction. Note that the point must be multiplied as a column
    vector from the right-hand side of the resulting matrix.

    Parameters
    ----------
    e : torch.Tensor
        East translation distances in meters.
        Shape is ``[number_of_heliostats]``.
    n : torch.Tensor
        North translation distances in meters.
        Shape is ``[number_of_heliostats]``.
    u : torch.Tensor
        Up translation distances in meters.
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Batched 4×4 translation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    if not (e.shape == u.shape == n.shape):
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

    ones = torch.ones_like(e, device=device)

    matrix = torch.zeros(e.shape[indices.heliostat_dimension], 4, 4, device=device)

    matrix[:, indices.e, indices.e] = ones
    matrix[:, indices.e, indices.transform_homogeneous] = e
    matrix[:, indices.n, indices.n] = ones
    matrix[:, indices.n, indices.transform_homogeneous] = n
    matrix[:, indices.u, indices.u] = ones
    matrix[:, indices.u, indices.transform_homogeneous] = u
    matrix[:, indices.transform_homogeneous, indices.transform_homogeneous] = ones

    return matrix


def perform_canting(
    canting_angles: torch.Tensor,
    data: torch.Tensor,
    inverse: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Perform canting (rotation) on data like surface points or surface normals.

    Parameters
    ----------
    canting_angles : torch.Tensor
        Canting angles.
        Shape is ``[number_of_surfaces, number_of_facets, 2, 4]``.
    data : torch.Tensor
        Data to be canted.
        Shape is ``[number_of_surfaces, number_of_facets, number_of_points_per_facet, 4]``.
    inverse : bool
        Indicates the direction of the rotation.
        Use ``inverse=False`` for canting and ``inverse=True`` for decanting (default is False).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The (de-)canted data.
        Shape is ``[number_of_surfaces, number_of_facets, number_of_points_per_facet, 4]``.
    """
    device = get_device(device=device)
    canting_angles = canting_angles.to(device)
    data = data.to(device)

    number_of_surfaces = data.shape[indices.heliostat_dimension]
    number_of_facets_per_surface = data.shape[indices.facet_dimension]

    rotation_matrix = torch.zeros(
        (number_of_surfaces, number_of_facets_per_surface, 4, 4),
        device=device,
        dtype=data.dtype,
    )

    # Extract ENU basis candidates from canting tensor (drop homogeneous component).
    e = canting_angles[:, :, indices.e, : indices.slice_fourth_dimension]
    n = canting_angles[:, :, indices.n, : indices.slice_fourth_dimension]

    # Build a numerically stable orthonormal basis:
    # 1) normalize e
    e = torch.nn.functional.normalize(e, dim=-1)
    # 2) u = normalize(e x n)
    u = torch.linalg.cross(e, n, dim=-1)
    u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)
    # 3) n_ortho = normalize(u x e)
    n_ortho = torch.linalg.cross(u, e, dim=-1)
    n_ortho = torch.nn.functional.normalize(n_ortho, dim=-1, eps=1e-8)

    # Fill rotation matrix columns with ENU basis vectors.
    rotation_matrix[:, :, : indices.slice_fourth_dimension, indices.e] = e
    rotation_matrix[:, :, : indices.slice_fourth_dimension, indices.n] = n_ortho
    rotation_matrix[:, :, : indices.slice_fourth_dimension, indices.u] = u
    rotation_matrix[
        :, :, indices.transform_homogeneous, indices.transform_homogeneous
    ] = 1.0

    # Data is represented as row vectors (..., 4); therefore:
    # - forward canting uses R^T
    # - inverse canting uses R
    if inverse:
        return data @ rotation_matrix
    return data @ rotation_matrix.mT
