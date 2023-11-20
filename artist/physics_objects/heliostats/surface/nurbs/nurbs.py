from typing import List, Optional, Tuple, Type, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch as th

def setup_nurbs_surface(
        degree_x: int,
        degree_y: int,
        num_control_points_x: int,
        num_control_points_y: int,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute uninitialized parameters for a NURBS surface with the desired properties on the given device.

    Parameters
    ----------
    degree_x : int
        Spline degree in x dimension.
    degree_y : int
        Spline degree in y dimension.
    num_control_points_x : int
        Number of control points in x dimension, (rows).
    num_control_points_y : int
        Number of control points in y dimension, (columns).
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The control points, control weights, and knots in x and y dimension.
    """
    next_degree_x = degree_x + 1
    next_degree_y = degree_y + 1
    assert num_control_points_x > degree_x, \
        f'need at least {next_degree_x} control points in x dimension'
    assert num_control_points_y > degree_y, \
        f'need at least {next_degree_y} control points in y dimension'

    control_points_shape = (num_control_points_x, num_control_points_y)
    control_points = th.empty(
        control_points_shape + (3,),
        device=device,
    )
    # to get b-splines, set these weights to all ones
    control_point_weights = th.empty(
        control_points_shape + (1,),
        device=device,
    )
    control_point_weights.clamp_(1e-8, th.finfo().max)

    knots_x = th.zeros(num_control_points_x + next_degree_x, device=device)
    # knots_x[:next_degree_x] = 0
    knots_x[next_degree_x:-next_degree_x] = 0.5
    knots_x[-next_degree_x:] = 1

    knots_y = th.zeros(num_control_points_y + next_degree_y, device=device)
    # knots_y[:next_degree_y] = 0
    knots_y[next_degree_y:-next_degree_y] = 0.5
    knots_y[-next_degree_y:] = 1
    return control_points, control_point_weights, knots_x, knots_y

def calc_normals_and_surface_slow(
        evaluation_points_x: torch.Tensor,
        evaluation_points_y: torch.Tensor,
        degree_x: int,
        degree_y: int,
        control_points: torch.Tensor,
        control_point_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return both the evaluation and normals of the given NURBS surface
    at the given evaluation points.

    Parameters
    ----------
    evaluation_points_x : torch.Tensor
        The evaluation points in x dimension.
    evaluation_points_y : torch.Tensor
        The evaluation points in y dimension. 
    degree_x : int
        The spline degree in x dimension.
    degree_y : int
        The spline degree in y dimension.
    control_points : torch.Tensor
        The control points.
    control_point_weights : torch.Tensor
        The weights of the control points. 
    knots_x : torch.Tensor
        List of numbers representing the knots in x dimension.
    knots_y : torch.Tensor
        List of numbers representing the knots in y dimension.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]:
        The surface points and the surface normals.
    """
    derivs = calc_derivs_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv=1,
    )
    cross_prod = th.cross(derivs[1][0], derivs[0][1])
    return (
        derivs[0][0],
        cross_prod / th.linalg.norm(cross_prod, dim=1).unsqueeze(-1),
    )

def calc_derivs_surface_slow(
        evaluation_points_x: torch.Tensor,
        evaluation_points_y: torch.Tensor,
        degree_x: int,
        degree_y: int,
        control_points: torch.Tensor,
        control_point_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        nth_deriv: int = 1,
) -> List[List[torch.Tensor]]:
    """Compute partial derivatives up to `nth_deriv` at the given
    evaluation points for the given NURBS surface.

    The resulting 4-D tensor `derivs` contains at `derivs[:, k, l]` the
    derivatives with respect to `evaluation_points_x` `k` times and
    `evaluation_points_y` `l` times.

    Parameters
    ----------
    evaluation_points_x : torch.Tensor
        The evaluation points in x dimension.
    evaluation_points_y : torch.Tensor
        The evaluation points in y dimension. 
    degree_x : int
        The spline degree in x dimension.
    degree_y : int
        The spline degree in y dimension.
    control_points : torch.Tensor
        The control points.
    control_point_weights : torch.Tensor
        The weights of the control points. 
    knots_x : torch.Tensor
        List of numbers representing the knots in x dimension.
    knots_y : torch.Tensor
        List of numbers representing the knots in y dimension.
    nth_deriv: int = 1,
        The nth derivative.
    
    Returns
    -------
    List[List[torch.Tensor]]
        The partial derivatives at the given evaluation points.
    """
    check_nurbs_surface_constraints(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )

    dtype = control_points.dtype
    device = control_points.device
    next_nth_deriv = nth_deriv + 1

    projected = project_control_points(control_points, control_point_weights)
    Swders = calc_bspline_derivs_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        projected,
        knots_x,
        knots_y,
        nth_deriv,
    )
    Aders = Swders[:, :, :, :-1]
    wders = Swders[:, :, :, -1]
    result = [
        [
            th.empty(Aders.shape[0], Aders.shape[-1], device=device)
            for _ in range(Aders.shape[1])
        ]
        for _ in range(Aders.shape[2])
    ]
    for k in th.arange(next_nth_deriv, device=device):
        for m in th.arange(next_nth_deriv - k, device=device):
            vs = Aders[:, k, m]
            for j in th.arange(1, m + 1, device=device):
                vs = vs - (
                    th.binomial(m.to(dtype), j.to(dtype))
                    * wders[:, 0, j].unsqueeze(-1)
                    * result[k][m - j]
                )
            for i in th.arange(1, k + 1, device=device):
                vs = vs - (
                    th.binomial(k.to(dtype), i.to(dtype))
                    * wders[:, i, 0].unsqueeze(-1)
                    * result[k - i][m]
                )
                vs2 = th.zeros_like(vs)
                for j in th.arange(1, m + 1, device=device):
                    vs2 += (
                        th.binomial(m.to(dtype), j.to(dtype))
                        * wders[:, i, j].unsqueeze(-1)
                        * result[k - i][m - j]
                    )
                vs = vs - th.binomial(k.to(dtype), i.to(dtype)) * vs2
            result[k][m] = vs / wders[:, 0, 0].unsqueeze(-1)
    return result


@th.no_grad()
def check_nurbs_surface_constraints(
        evaluation_points_x: torch.Tensor,
        evaluation_points_y: torch.Tensor,
        degree_x: int,
        degree_y: int,
        control_points: torch.Tensor,
        control_point_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> None:
    """
    Assert that NURBS constraints are fulfilled for evaluating the given surface.

    Parameters
    ----------
    evaluation_points_x : torch.Tensor
        The evaluation points in x dimension.
    evaluation_points_y : torch.Tensor
        The evaluation points in y dimension. 
    degree_x : int
        The spline degree in x dimension.
    degree_y : int
        The spline degree in y dimension.
    control_points : torch.Tensor
        The control points.
    control_point_weights : torch.Tensor
        The weights of the control points. 
    knots_x : torch.Tensor
        List of numbers representing the knots in x dimension.
    knots_y : torch.Tensor
        List of numbers representing the knots in y dimension.
    """
    next_degree_x = degree_x + 1
    next_degree_y = degree_y + 1
    assert control_points.shape[-1] == 3, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert control_points.ndim == 3, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert (control_point_weights > 0).all(), \
        'control point weights must be greater than zero'
    assert (knots_x[:next_degree_x] == 0).all(), \
        f'first {next_degree_x} knots must be zero'
    assert (knots_x[control_points.shape[0]:] == 1).all(), \
        f'last {next_degree_x} knots must be one'
    assert (knots_x.sort().values == knots_x).all(), \
        'knots must be ordered monotonically increasing in value'
    assert (knots_y[:next_degree_y] == 0).all(), \
        f'first {next_degree_y} knots must be zero'
    assert (knots_y[control_points.shape[1]:] == 1).all(), \
        f'last {next_degree_y} knots must be one'
    assert (knots_y.sort().values == knots_y).all(), \
        'knots must be ordered monotonically increasing in value'
    assert evaluation_points_x.shape == evaluation_points_y.shape, \
        "evaluation point shapes don't match"

def project_control_points(
        control_points: torch.Tensor,
        control_point_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Project the given n-D control points with their weights into (n + 1)-D space.

    Parameters
    ----------
    control_points : torch.Tensor
        The control points.
    control_point_weights : torch.Tensor
        The weights of the control points.
    
    Returns
    -------
    torch.Tensor
        The projection.
    """
    projected = control_point_weights * control_points
    projected = th.cat([projected, control_point_weights], dim=-1)
    return projected

def calc_bspline_derivs_surface_slow(
        evaluation_points_x: torch.Tensor,
        evaluation_points_y: torch.Tensor,
        degree_x: int,
        degree_y: int,
        control_points: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        nth_deriv: int = 1,
) -> torch.Tensor:
    """
    Compute partial derivatives up to `nth_deriv` at the given evaluation points for the given B-spline surface.

    The resulting 4-D tensor `derivs` contains at `derivs[:, k, l]` the
    derivatives with respect to `evaluation_points_x` `k` times and
    `evaluation_points_y` `l` times.

    Parameters
    ----------
    evaluation_points_x : torch.Tensor
        The evaluation points in x dimension.
    evaluation_points_y : torch.Tensor
        The evaluation points in y dimension. 
    degree_x : int
        The spline degree in x dimension.
    degree_y : int
        The spline degree in y dimension.
    control_points : torch.Tensor
        The control points.
    knots_x : torch.Tensor
        List of numbers representing the knots in x dimension.
    knots_y : torch.Tensor
        List of numbers representing the knots in y dimension.
    nth_deriv: int = 1,
        The nth derivative.
    
    Returns
    -------
    List[List[torch.Tensor]]
        The partial derivatives at the given evaluation points.
    """
    device = control_points.device
    num_evaluation_points = len(evaluation_points_x)
    next_nth_deriv = nth_deriv + 1
    next_degree_x = degree_x + 1
    next_degree_y = degree_y + 1
    result = th.empty(
        (
            num_evaluation_points,
            next_nth_deriv,
            next_nth_deriv,
            control_points.shape[-1],
        ),
        device=device,
    )
    du = min(nth_deriv, degree_x)
    for k in range(next_degree_x, next_nth_deriv):
        for j in range(next_nth_deriv - k):
            result[:, k, j] = 0
    dv = min(nth_deriv, degree_y)
    for j in range(next_degree_y, next_nth_deriv):
        for k in range(next_nth_deriv - j):
            result[:, k, j] = 0

    num_control_points_x = control_points.shape[0]
    spans_x = find_span(
        evaluation_points_x, degree_x, num_control_points_x, knots_x)
    basis_derivs_x = calc_basis_derivs_slow(
        evaluation_points_x, spans_x, degree_x, knots_x, du)

    num_control_points_y = control_points.shape[1]
    spans_y = find_span(
        evaluation_points_y, degree_y, num_control_points_y, knots_y)
    basis_derivs_y = calc_basis_derivs_slow(
        evaluation_points_y, spans_y, degree_y, knots_y, dv)

    tmp = [
        th.empty(
            (num_evaluation_points, control_points.shape[-1]),
            device=device,
        )
        for _ in range(next_degree_y)
    ]
    spanmdegree_x = spans_x - degree_x
    spanmdegree_y = spans_y - degree_y
    for k in range(du + 1):
        for s in range(next_degree_y):
            tmp[s] = th.zeros_like(tmp[s])
            for r in range(next_degree_x):
                tmp[s] += (
                    basis_derivs_x[k][r].unsqueeze(-1)
                    * control_points[spanmdegree_x + r, spanmdegree_y + s]
                )
        dd = min(nth_deriv - k, dv)
        for j in range(dd + 1):
            result[:, k, j] = 0
            for s in range(next_degree_y):
                result[:, k, j] += \
                    basis_derivs_y[j][s].unsqueeze(-1) * tmp[s]
    return result

def find_span(
        evaluation_points: torch.Tensor,
        degree: int,
        num_control_points: int,
        knots: torch.Tensor,
) -> torch.Tensor:
    """
    For each evaluation point, compute the span in which it lies.
    
    Parameters
    ----------
    evaluation_points : torch.Tensor
        The evaluation points.
    degree : int
        The spline degree.
    num_control_points : int
        The number of control points.
    knots : torch.Tensor
        The knots.

    Returns
    -------
    torch.Tensor
        The spans in which the evaluation points lie.
    """
    result = th.empty(
        len(evaluation_points),
        dtype=th.int64,
        device=knots.device,
    )
    not_upper_span_indices = \
        evaluation_points != knots[num_control_points]
    result[~not_upper_span_indices] = num_control_points - 1
    spans = th.searchsorted(
        knots,
        evaluation_points[not_upper_span_indices],
        right=True,
    ) - 1
    result[not_upper_span_indices] = spans
    return result

def calc_basis_derivs_slow(
        evaluation_points: torch.Tensor,
        span: torch.Tensor,
        degree: int,
        knots: torch.Tensor,
        nth_deriv: int = 1,
) -> List[List[torch.Tensor]]:
    """
    Compute the first `nth_deriv` derivatives for the basis functions
    applied to the given evaluation points. The k-th derivative is at
    index k, 0 <= k <= `nth_deriv`.

    Parameters
    ----------
    evaluation_points : torch.Tensor
        The evaluation points.
    span : torch.Tensor
        The spans in which the evaluation points lie.
    degree : int
        The spline degree
    knots : torch.Tensor
        The knots.
    nth_deriv : int = 1
        The nth derivative.

    Returns
    -------
    The first nth derivatives.
    """
    device = knots.device
    num_evaluation_points = len(evaluation_points)
    next_span = span + 1
    next_degree = degree + 1
    next_nth_deriv = nth_deriv + 1
    ndu = [
        [
            th.empty(
                (num_evaluation_points,),
                device=device,
            )
            for _ in range(next_degree)
        ]
        for _ in range(next_degree)
    ]
    ndu[0][0] = th.ones_like(ndu[0][0])
    left = [
        th.empty((num_evaluation_points,), device=device)
        for _ in range(next_degree)
    ]
    right = [
        th.empty((num_evaluation_points,), device=device)
        for _ in range(next_degree)
    ]
    for j in range(1, next_degree):
        left[j] = evaluation_points - knots[next_span - j]
        right[j] = knots[span + j] - evaluation_points
        saved = th.zeros(num_evaluation_points, device=device)
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            tmp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        ndu[j][j] = saved

    ders = [
        [
            th.empty((num_evaluation_points,), device=device)
            for _ in range(next_degree)
        ]
        for _ in range(next_nth_deriv)
    ]
    for j in range(next_degree):
        ders[0][j] = ndu[j][degree]
    a = [
        [
            th.empty((num_evaluation_points,), device=device)
            for _ in range(next_degree)
        ]
        for _ in range(2)
    ]
    for r in range(next_degree):
        s1 = 0
        s2 = 1
        a[0][0] = th.ones_like(a[0][0])
        for k in range(1, next_nth_deriv):
            d = th.zeros(num_evaluation_points, device=device)
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = degree - r
            for j in range(j1, j2 + 1):
                a[s2][j] = (
                    (a[s1][j] - a[s1][j - 1])
                    / ndu[pk + 1][rk + j]
                )
                d += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += a[s2][k] * ndu[r][pk]
            ders[k][r] = d
            j = s1
            s1 = s2
            s2 = j

    r = degree
    for k in range(1, next_nth_deriv):
        for j in range(next_degree):
            ders[k][j] *= r
        r *= degree - k
    return ders

