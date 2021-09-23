import matplotlib.pyplot as plt
import torch as th

EPS = 1e-7


def setup_nurbs(degree, num_control_points, device):
    assert num_control_points > degree, \
        f'need at least {degree + 1} control points'
    control_points = th.empty((num_control_points, 2), device=device)
    # to get b-splines, set these weights to all ones
    control_point_weights = th.empty((num_control_points, 1), device=device)
    control_point_weights.clamp_(1e-8, th.finfo().max)
    next_degree = degree + 1
    knots = th.empty(num_control_points + next_degree, device=device)
    knots[:next_degree] = 0
    knots[next_degree:-next_degree] = 0.5
    knots[-next_degree:] = 1
    return control_points, control_point_weights, knots


def find_span(evaluation_points, degree, num_control_points, knots):
    after_num_control_points = num_control_points + 1
    result = th.empty(
        len(evaluation_points),
        dtype=th.int64,
        device=knots.device,
    )
    not_upper_span_indices = \
        evaluation_points != knots[after_num_control_points]
    result[~not_upper_span_indices] = num_control_points
    spans = th.searchsorted(
        knots,
        evaluation_points[not_upper_span_indices],
        right=True,
    ) - 1
    result[not_upper_span_indices] = spans
    return result


def get_basis(evaluation_points, span, degree, knots):
    device = knots.device
    num_evaluation_points = len(evaluation_points)
    next_degree = degree + 1
    next_span = span + 1
    basis_values = th.empty(
        (num_evaluation_points, next_degree),
        device=device,
    )
    basis_values[:, 0] = 1
    left = th.empty((num_evaluation_points, next_degree), device=device)
    right = th.empty((num_evaluation_points, next_degree), device=device)
    for j in range(1, next_degree):
        left[:, j] = evaluation_points - knots[next_span - j]
        right[:, j] = knots[span + j] - evaluation_points
        saved = th.zeros(num_evaluation_points, device=device)
        for r in range(j):
            divisor = right[:, r + 1] + left[:, j - r]
            tmp = th.empty_like(divisor)
            nonzero_indices = divisor != 0
            tmp[~nonzero_indices] = 0
            tmp[nonzero_indices] = (
                basis_values[nonzero_indices, r]
                / divisor[nonzero_indices]
            )
            basis_values[:, r] = saved + right[:, r + 1] * tmp
            saved = left[:, j - r] * tmp
        basis_values[:, j] = saved
    basis_values[span == len(knots) - degree - 1, -1] = 1
    return basis_values


def get_all_basis(evaluation_point, span, degree, knots):
    # FIXME
    device = knots.device
    next_degree = degree + 1
    next_span = span + 1
    basis_values = th.empty((next_degree,) * 2, device=device)
    basis_values[0, 0] = 1
    print(basis_values)
    left = th.empty(next_degree, device=device)
    right = th.empty(next_degree, device=device)
    for j in range(1, next_degree):
        left[j] = evaluation_point - knots[next_span - j]
        right[j] = knots[span + j] - evaluation_point
        saved = 0
        for r in range(j):
            tmp = basis_values[r, r] / (right[r + 1] + left[j - r])
            # correct(er) results when r and j indices are swapped here
            # however, does not correspond to algorithm; so something's
            # probably wrong
            basis_values[r, j] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        basis_values[j, j] = saved
    print(basis_values)
    return basis_values


def calc_basis_derivs(
        evaluation_points,
        span,
        degree,
        knots,
        nth_deriv=1,
):
    device = knots.device
    num_evaluation_points = len(evaluation_points)
    next_span = span + 1
    next_degree = degree + 1
    next_nth_deriv = nth_deriv + 1
    ndu = th.empty(
        (num_evaluation_points, next_degree, next_degree),
        device=device,
    )
    ndu[:, 0, 0] = 1
    left = th.empty((num_evaluation_points, next_degree), device=device)
    right = th.empty((num_evaluation_points, next_degree), device=device)
    for j in range(1, next_degree):
        left[:, j] = evaluation_points - knots[next_span - j]
        right[:, j] = knots[span + j] - evaluation_points
        saved = 0
        for r in range(j):
            ndu[:, j, r] = right[:, r + 1] + left[:, j - r]
            tmp = ndu[:, r, j - 1] / ndu[:, j, r]
            ndu[:, r, j] = saved + right[:, r + 1] * tmp
            saved = left[:, j - r] * tmp
        ndu[:, j, j] = saved

    ders = th.empty(
        (num_evaluation_points, next_nth_deriv, next_degree),
        device=device,
    )
    for j in range(next_degree):
        ders[:, 0, j] = ndu[:, j, degree]
    a = th.empty((num_evaluation_points, 2, next_degree), device=device)
    for r in range(next_degree):
        s1 = 0
        s2 = 1
        a[:, 0, 0] = 1
        for k in range(1, next_nth_deriv):
            d = 0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[:, s2, 0] = a[:, s1, 0] / ndu[:, pk + 1, rk]
                d = a[:, s2, 0] * ndu[:, rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = degree - r
            for j in range(j1, j2 + 1):
                a[:, s2, j] = (
                    (a[:, s1, j] - a[:, s1, j - 1])
                    / ndu[:, pk + 1, rk + j]
                )
                d += a[:, s2, j] * ndu[:, rk + j, pk]
            if r <= pk:
                a[:, s2, k] = -a[:, s1, k - 1] / ndu[:, pk + 1, r]
                d += a[:, s2, k] * ndu[:, r, pk]
            ders[:, k, r] = d
            j = s1
            s1 = s2
            s2 = j

    r = degree
    for k in range(1, next_nth_deriv):
        for j in range(next_degree):
            ders[:, k, j] *= r
        r *= degree - k
    return ders


def calc_basis_derivs_slow(
        evaluation_points,
        span,
        degree,
        knots,
        nth_deriv=1,
):
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
        saved = 0
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
            d = 0
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
                d += a[s2, j] * ndu[rk + j][pk]
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


def project_control_points(control_points, control_point_weights):
    projected = control_point_weights * control_points
    projected = th.cat([projected, control_point_weights], dim=-1)
    return projected


def evaluate_nurbs(
        evaluation_points,
        degree,
        control_points,
        control_point_weights,
        knots,
):
    next_degree = degree + 1
    num_control_points = len(control_points)
    assert control_points.shape[-1] == 2, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert control_points.ndim == 2, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert (control_point_weights > 0).all(), \
        'control point weights must be greater than zero'
    assert (knots[:next_degree] == 0).all(), \
        f'first {next_degree} knots must be zero'
    assert (knots[num_control_points] == 1).all(), \
        f'last {next_degree} knots must be one'
    assert (knots.sort().values == knots).all(), \
        'knots must be ordered monotonically increasing in value'

    projected = project_control_points(control_points, control_point_weights)
    spans = find_span(evaluation_points, degree, num_control_points, knots)
    spansmdeg = th.clamp(spans, max=num_control_points - 1) - degree
    basis_values = get_basis(evaluation_points, spans, degree, knots)
    Cw = th.zeros(
        (len(evaluation_points), projected.shape[-1]),
        device=control_points.device,
    )
    for j in range(next_degree):
        Cw += basis_values[:, j] * projected[spansmdeg + j]
    return Cw[:, :-1] / Cw[:, -1]


def calc_bspline_derivs(
        evaluation_point,
        degree,
        control_points,
        knots,
        nth_deriv=1,
):
    device = control_points.device
    next_degree = degree + 1
    next_nth_deriv = nth_deriv + 1
    num_control_points = len(control_points) - 1
    du = min(nth_deriv, degree)
    result = th.empty(
        (next_nth_deriv, control_points.shape[-1]), device=device)
    for k in range(next_degree, next_nth_deriv):
        result[k] = 0
    span = find_span(evaluation_point, degree, num_control_points, knots)
    basis_derivs = calc_basis_derivs(evaluation_point, span, degree, knots, du)
    spanmdeg = th.clamp(span, num_control_points) - degree
    for k in range(du + 1):
        result[k] = 0
        for j in range(next_degree):
            result[k] += basis_derivs[k, j] * control_points[spanmdeg + j]
    # the k-th derivative is at index k, 0 <= k <= nth_deriv
    return result


def _calc_bspline_deriv_cpts(
        r1,
        r2,
        degree,
        control_points,
        knots,
        nth_deriv=1,
):
    device = control_points.device
    next_nth_deriv = nth_deriv + 1
    next_degree = degree + 1
    r = r2 - r1
    result = th.empty(
        (nth_deriv + 1, r + 1, control_points.shape[-1]),
        device=device,
    )
    # print(result.shape)
    # print(control_points.shape)
    for i in range(r + 1):
        result[0, i] = control_points[r1 + i]
    for k in range(1, next_nth_deriv):
        tmp = degree - k + 1
        for i in range(r - k + 1):
            result[k, i] = (
                tmp * (result[k - 1, i + 1] - result[k - 1, i])
                / (knots[r1 + i + next_degree] - knots[r1 + i + k])
            )
    return result


def calc_bspline_derivs_2(
        evaluation_point,
        degree,
        control_points,
        knots,
        nth_deriv=1,
):
    device = control_points.device
    next_degree = degree + 1
    next_nth_deriv = nth_deriv + 1
    num_control_points = len(control_points) - 1
    du = min(nth_deriv, degree)
    result = th.empty(
        (next_nth_deriv, control_points.shape[-1]), device=device)
    for k in range(next_degree, next_nth_deriv):
        result[k] = 0
    span = find_span(evaluation_point, degree, num_control_points, knots)
    all_basis_values = get_all_basis(evaluation_point, span, degree, knots)
    pk = _calc_bspline_deriv_cpts(
        span - degree, span, degree, control_points, knots, du)
    for k in range(du + 1):
        result[k] = 0
        for j in range(next_degree - k):
            result[k] += all_basis_values[j, degree - k] * pk[k, j]
    # the k-th derivative is at index k, 0 <= k <= nth_deriv
    return result


def calc_derivs(
        evaluation_point,
        degree,
        control_points,
        control_point_weights,
        knots,
        nth_deriv=1,
):
    device = control_points.device
    next_nth_deriv = nth_deriv + 1
    projected = project_control_points(control_points, control_point_weights)
    Cwders = calc_bspline_derivs(
        evaluation_point, degree, projected, knots, nth_deriv)
    Aders = Cwders[:, :-1]
    wders = Cwders[:, -1]
    result = th.empty_like(Aders)
    for k in th.arange(next_nth_deriv, device=device):
        v = Aders[k]
        for i in th.arange(1, k + 1, device=device):
            v -= th.binomial(k.float(), i.float()) * wders[i] * result[k - i]
        result[k] = v / wders[0]
    return result


# def get_basis(evaluation_point, control_point_index, degree, knots):
#     next_control_point_index = control_point_index + 1
#     prev_degree = degree - 1
#     if degree == 0:
#         return int(
#             knots[control_point_index]
#             <= evaluation_point
#             < knots[next_control_point_index]
#         )
#     return (
#         (
#             f_rising(evaluation_point, control_point_index, degree, knots)
#             * get_basis(
#                 evaluation_point, control_point_index, prev_degree, knots)
#         )
#         + (
#             g_falling(
#                 evaluation_point, next_control_point_index, degree, knots)
#             * get_basis(
#                 evaluation_point,
#                 next_control_point_index,
#                 prev_degree,
#                 knots,
#             )
#         )
#     )


# def f_rising(evaluation_point, control_point_index, degree, knots):
#     curr_knot = knots[control_point_index]
#     dividend = evaluation_point - curr_knot
#     divisor = knots[control_point_index + degree] - curr_knot
#     if divisor == 0 == dividend:
#         return 0
#     return dividend / divisor


# def g_falling(evaluation_point, control_point_index, degree, knots):
#     later_knot = knots[control_point_index + degree]
#     dividend = later_knot - evaluation_point
#     divisor = later_knot - knots[control_point_index]
#     if divisor == 0 == dividend:
#         return 0
#     # equal to 1 - f_rising([...])
#     return dividend / divisor


# def calc_projected_weights(
#         evaluation_point,
#         degree,
#         control_point_weights,
#         knots,
# ):
#     device = control_point_weights.device

#     span = find_span(
#         evaluation_point, len(control_point_weights), degree, knots)
#     basis_values = get_basis(evaluation_point, span, degree, knots)
#     # control_point_indices = th.arange(
#     #     len(control_point_weights), device=device)
#     # basis_values = th.tensor([
#     #     get_basis(evaluation_point, control_point_index, degree, knots)
#     #     for control_point_index in control_point_indices
#     # ], dtype=th.float32, device=device)
#     print(basis_values)

#     projected_weights = basis_values * control_point_weights
#     print(projected_weights)
#     return projected_weights


# def rational_basis(
#         evaluation_point,
#         degree,
#         control_point_weights,
#         knots,
# ):
#     projected_weights = calc_projected_weights(
#         evaluation_point,
#         degree,
#         control_point_weights,
#         knots,
#     )
#     return projected_weights / th.sum(projected_weights)


# def evaluate_nurbs(
#         evaluation_point,
#         degree,
#         control_points,
#         control_point_weights,
#         knots,
# ):
#     # assert control_points.ndim == 1, (
#     #     "please use another evaluation function for "
#     #     "this NURBS' dimensionality"
#     # )
#     # assert (control_point_weights > 0).all(), \
#     #     'control point weights must be greater than zero'
#     # assert (knots[:degree + 1] == 0).all(), \
#     #     f'first {degree + 1} knots must be zero'
#     # assert (knots[len(control_points)] == 1).all(), \
#     #     f'last {degree + 1} knots must be one'
#     # assert (knots.sort().values == knots).all(), \
#     #     'knots must be ordered monotonically increasing in value'

#     rational_basis_values = rational_basis(
#         evaluation_point, degree, control_point_weights, knots)
#     return th.sum(rational_basis_values * control_points)

#     # device = control_point_weights.device

#     # control_point_indices = th.arange(
#     #     len(control_point_weights), device=device)
#     # basis_values = th.tensor([
#     #     get_basis(evaluation_point, control_point_index, degree, knots)
#     #     for control_point_index in control_point_indices
#     # ], dtype=th.float32, device=device)

#     # projected_weights = basis_values * control_point_weights
#     # return (
#     #     th.sum(projected_weights * control_points)
#     #     / th.sum(projected_weights)
#     # )


class NURBSCurve:
    def __init__(self, degree, control_points, control_point_weights, knots):
        self.degree = degree
        self.control_points = control_points
        self.control_point_weights = control_point_weights
        self.knots = knots

    @classmethod
    def create_empty(cls, degree, num_control_points, device):
        control_points, control_point_weights, knots = setup_nurbs(
            degree, num_control_points, device)
        return cls(degree, control_points, control_point_weights, knots)

    def find_span(self, evaluation_point, num_control_points=None):
        if num_control_points is None:
            num_control_points = len(self.control_points)
        return self._find_span(
            evaluation_point, self.degree, num_control_points, self.knots)

    def get_basis(self, evaluation_point, span=None):
        if span is None:
            span = self.find_span(evaluation_point)
        return get_basis(evaluation_point, span, self.degree, self.knots)

    def get_all_basis(self, evaluation_point, span=None):
        if span is None:
            span = self.find_span(evaluation_point)
        return get_all_basis(evaluation_point, span, self.degree, self.knots)

    def calc_basis_derivs(self, evaluation_point, span=None, nth_deriv=1):
        if span is None:
            span = self.find_span(evaluation_point)
        return calc_basis_derivs(
            evaluation_point, span, self.degree, self.knots, nth_deriv)

    def project_control_points(self):
        return project_control_points(
            self.control_points, self.control_point_weights)

    def evaluate(self, evaluation_point):
        return evaluate_nurbs(
            evaluation_point,
            self.degree,
            self.control_points,
            self.control_point_weights,
            self.knots,
        )

    def calc_bspline_derivs(self, evaluation_point, nth_deriv=1):
        return calc_bspline_derivs(
            evaluation_point,
            self.degree,
            self.control_points,
            self.knots,
            nth_deriv,
        )

    def calc_bspline_derivs_2(self, evaluation_point, nth_deriv=1):
        return calc_bspline_derivs_2(
            evaluation_point,
            self.degree,
            self.control_points,
            self.knots,
            nth_deriv,
        )

    def calc_derivs(self, evaluation_point, nth_deriv=1):
        return calc_derivs(
            evaluation_point,
            self.degree,
            self.control_points,
            self.control_point_weights,
            self.knots,
            nth_deriv,
        )


def setup_nurbs_surface(
        degree_x,
        degree_y,
        num_control_points_x,
        num_control_points_y,
        device,
):
    next_degree_x = degree_x + 1
    next_degree_y = degree_y + 1
    assert num_control_points_x > degree_x, \
        f'need at least {next_degree_x} control points in x direction'
    assert num_control_points_y > degree_y, \
        f'need at least {next_degree_y} control points in y direction'

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


def check_nurbs_constraints(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
):
    assert control_points.shape[-1] == 3, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert control_points.ndim == 3, \
        "please use another evaluation function for this NURBS' dimensionality"
    assert (control_point_weights > 0).all(), \
        'control point weights must be greater than zero'
    assert (knots_x[:degree_x + 1] == 0).all(), \
        f'first {degree_x + 1} knots must be zero'
    assert (knots_x[control_points.shape[0]] == 1).all(), \
        f'last {degree_x + 1} knots must be one'
    assert (knots_x.sort().values == knots_x).all(), \
        'knots must be ordered monotonically increasing in value'
    assert (knots_y[:degree_y + 1] == 0).all(), \
        f'first {degree_y + 1} knots must be zero'
    assert (knots_y[control_points.shape[1]] == 1).all(), \
        f'last {degree_y + 1} knots must be one'
    assert (knots_y.sort().values == knots_y).all(), \
        'knots must be ordered monotonically increasing in value'
    assert evaluation_points_x.shape == evaluation_points_x.shape, \
        "evaluation point shapes don't match"


def evaluate_nurbs_surface_at_spans(
        num_evaluation_points,
        spans_x,
        spans_y,
        basis_values_x,
        basis_values_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
):
    device = control_points.device
    projected = project_control_points(control_points, control_point_weights)
    tmp = th.empty(
        (num_evaluation_points, degree_y + 1, projected.shape[-1]),
        device=device,
    )
    spansmdeg_x = th.clamp(spans_x, max=control_points.shape[0] - 1) - degree_x
    spansmdeg_y = th.clamp(spans_y, max=control_points.shape[1] - 1) - degree_y
    for j in range(degree_y + 1):
        tmp[:, j] = 0
        for k in range(degree_x + 1):
            tmp[:, j] += (
                basis_values_x[:, k].unsqueeze(-1)
                * projected[spansmdeg_x + k, spansmdeg_y + j]
            )
    Sw = th.zeros((num_evaluation_points, projected.shape[-1]), device=device)
    for j in range(degree_y + 1):
        Sw += basis_values_y[:, j].unsqueeze(-1) * tmp[:, j]
    return Sw[:, :-1] / Sw[:, -1].unsqueeze(-1)


def evaluate_nurbs_surface_flex(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
):
    check_nurbs_constraints(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )

    num_evaluation_points = len(evaluation_points_x)
    num_control_points_x = control_points.shape[0]
    spans_x = find_span(
        evaluation_points_x, degree_x, num_control_points_x, knots_x)
    basis_values_x = get_basis(evaluation_points_x, spans_x, degree_x, knots_x)

    num_control_points_y = control_points.shape[1]
    spans_y = find_span(
        evaluation_points_y, degree_y, num_control_points_y, knots_y)
    basis_values_y = get_basis(evaluation_points_y, spans_y, degree_y, knots_y)

    return evaluate_nurbs_surface_at_spans(
        num_evaluation_points,
        spans_x,
        spans_y,
        basis_values_x,
        basis_values_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
    )


def calc_bspline_derivs_surface(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        knots_x,
        knots_y,
        nth_deriv=1,
):
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
    basis_derivs_x = calc_basis_derivs(
        evaluation_points_x, spans_x, degree_x, knots_x, du)

    num_control_points_y = control_points.shape[1]
    spans_y = find_span(
        evaluation_points_y, degree_y, num_control_points_y, knots_y)
    basis_derivs_y = calc_basis_derivs(
        evaluation_points_y, spans_y, degree_y, knots_y, dv)

    tmp = th.empty(
        (num_evaluation_points, next_degree_y, control_points.shape[-1]),
        device=device,
    )
    spanmdegree_x = th.clamp(spans_x, max=num_control_points_x - 1) - degree_x
    spanmdegree_y = th.clamp(spans_y, max=num_control_points_y - 1) - degree_y
    for k in range(du + 1):
        for s in range(next_degree_y):
            tmp[:, s] = 0
            for r in range(next_degree_x):
                tmp[:, s] += (
                    basis_derivs_x[:, k, r].unsqueeze(-1)
                    * control_points[spanmdegree_x + r, spanmdegree_y + s]
                )
        dd = min(nth_deriv - k, dv)
        for j in range(dd + 1):
            result[:, k, j] = 0
            for s in range(next_degree_y):
                result[:, k, j] += \
                    basis_derivs_y[:, j, s].unsqueeze(-1) * tmp[:, s]
    return result


def calc_bspline_derivs_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        knots_x,
        knots_y,
        nth_deriv=1,
):
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
    spanmdegree_x = th.clamp(spans_x, max=num_control_points_x - 1) - degree_x
    spanmdegree_y = th.clamp(spans_y, max=num_control_points_y - 1) - degree_y
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


def calc_derivs_surface(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv=1,
):
    """Return partial derivatives up to `nth_deriv` at the given
    evaluation points for the given NURBS.

    The resulting 4-D tensor `derivs` contains at `derivs[:, k, l]` the
    derivatives with respect to `evaluation_points_x` `k` times and
    `evaluation_points_y` `l` times.
    """
    check_nurbs_constraints(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )

    device = control_points.device
    next_nth_deriv = nth_deriv + 1
    projected = project_control_points(control_points, control_point_weights)
    Swders = calc_bspline_derivs_surface(
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
    result = th.empty_like(Aders)
    for k in th.arange(next_nth_deriv, device=device):
        for m in th.arange(next_nth_deriv - k, device=device):
            vs = Aders[:, k, m]
            for j in th.arange(1, m + 1, device=device):
                vs -= (
                    th.binomial(m.float(), j.float())
                    * wders[:, 0, j].unsqueeze(-1)
                    * result[:, k, m - j]
                )
            for i in th.arange(1, k + 1, device=device):
                vs -= (
                    th.binomial(k.float(), i.float())
                    * wders[:, i, 0].unsqueeze(-1)
                    * result[:, k - i, m]
                )
                vs2 = th.zeros_like(vs)
                for j in th.arange(1, m + 1, device=device):
                    vs2 += (
                        th.binomial(m.float(), j.float())
                        * wders[:, i, j].unsqueeze(-1)
                        * result[:, k - i, m - j]
                    )
                vs -= th.binomial(k.float(), i.float()) * vs2
            result[:, k, m] = vs / wders[:, 0, 0].unsqueeze(-1)
    return result


def calc_derivs_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv=1,
):
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
                    th.binomial(m.float(), j.float())
                    * wders[:, 0, j].unsqueeze(-1)
                    * result[k][m - j]
                )
            for i in th.arange(1, k + 1, device=device):
                vs = vs - (
                    th.binomial(k.float(), i.float())
                    * wders[:, i, 0].unsqueeze(-1)
                    * result[k - i][m]
                )
                vs2 = th.zeros_like(vs)
                for j in th.arange(1, m + 1, device=device):
                    vs2 += (
                        th.binomial(m.float(), j.float())
                        * wders[:, i, j].unsqueeze(-1)
                        * result[k - i][m - j]
                    )
                vs = vs - th.binomial(k.float(), i.float()) * vs2
            result[k][m] = vs / wders[:, 0, 0].unsqueeze(-1)
    return result


def calc_normals_surface(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
):
    derivs = calc_derivs_surface(
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
    cross_prod = th.cross(derivs[:, 1, 0], derivs[:, 0, 1])
    return cross_prod / th.linalg.norm(cross_prod, dim=1).unsqueeze(-1)


def calc_normals_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
):
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
    return cross_prod / th.linalg.norm(cross_prod, dim=1).unsqueeze(-1)


def calc_normals_and_surface_slow(
        evaluation_points_x,
        evaluation_points_y,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
):
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


def plot_surface(
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        step_granularity_x=0.02,
        step_granularity_y=0.02,
        show_plot=True,
):
    device = control_points.device
    xs = th.arange(0, 1, step_granularity_x, device=device)
    ys = th.arange(0, 1, step_granularity_y, device=device)
    xs = th.hstack([xs, th.tensor(1 - EPS, device=device)])
    ys = th.hstack([ys, th.tensor(1 - EPS, device=device)])

    eval_points = th.cartesian_prod(xs, ys)
    res = evaluate_nurbs_surface_flex(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )
    res = res.reshape((len(xs), len(ys)) + res.shape[1:])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.3,
        label='control_points',
    )
    ax.plot_wireframe(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.3,
    )
    ax.plot_surface(
        res[:, :, 0].detach().cpu().numpy(),
        res[:, :, 1].detach().cpu().numpy(),
        res[:, :, 2].detach().cpu().numpy(),
        cmap='plasma',
        alpha=0.8,
    )
    if show_plot:
        plt.show()
    return fig, ax


def plot_surface_derivs(
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        step_granularity_x=0.02,
        step_granularity_y=0.02,
        nth_deriv=1,
        show_plot=True,
        plot_normals=None,
):
    if plot_normals is None:
        plot_normals = nth_deriv == 1
    device = control_points.device
    xs = th.arange(0, 1, step_granularity_x, device=device)
    ys = th.arange(0, 1, step_granularity_y, device=device)
    xs = th.hstack([xs, th.tensor(1 - EPS, device=device)])
    ys = th.hstack([ys, th.tensor(1 - EPS, device=device)])

    eval_points = th.cartesian_prod(xs, ys)
    res = calc_derivs_surface(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv,
    )
    res = res.reshape((len(xs), len(ys)) + res.shape[1:])
    if plot_normals:
        normals = calc_normals_surface(
            eval_points[:, 0],
            eval_points[:, 1],
            degree_x,
            degree_y,
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
        )
        normals = normals.reshape((len(xs), len(ys)) + normals.shape[1:])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
        label='control_points',
    )
    ax.plot_wireframe(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
    )
    ax.plot_surface(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        cmap='plasma',
        alpha=0.3,
    )
    ax.quiver(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        res[:, :, 1, 0, 0].detach().cpu().numpy(),
        res[:, :, 1, 0, 1].detach().cpu().numpy(),
        res[:, :, 1, 0, 2].detach().cpu().numpy(),
        length=0.05,
        alpha=0.8,
        label='dS/dx',
    )
    ax.quiver(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        res[:, :, 0, 1, 0].detach().cpu().numpy(),
        res[:, :, 0, 1, 1].detach().cpu().numpy(),
        res[:, :, 0, 1, 2].detach().cpu().numpy(),
        length=0.05,
        color='red',
        alpha=0.8,
        label='dS/dy',
    )
    if plot_normals:
        ax.quiver(
            res[:, :, 0, 0, 0].detach().cpu().numpy(),
            res[:, :, 0, 0, 1].detach().cpu().numpy(),
            res[:, :, 0, 0, 2].detach().cpu().numpy(),
            normals[:, :, 0].detach().cpu().numpy(),
            normals[:, :, 1].detach().cpu().numpy(),
            normals[:, :, 2].detach().cpu().numpy(),
            length=0.05,
            color='green',
            alpha=0.8,
            label='normals',
        )
    ax.legend()
    if show_plot:
        plt.show()
    return fig, ax


def plot_surface_derivs_slow(
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        step_granularity_x=0.02,
        step_granularity_y=0.02,
        nth_deriv=1,
        show_plot=True,
        plot_normals=None,
):
    if plot_normals is None:
        plot_normals = nth_deriv == 1
    device = control_points.device
    xs = th.arange(0, 1, step_granularity_x, device=device)
    ys = th.arange(0, 1, step_granularity_y, device=device)
    xs = th.hstack([xs, th.tensor(1 - EPS, device=device)])
    ys = th.hstack([ys, th.tensor(1 - EPS, device=device)])

    eval_points = th.cartesian_prod(xs, ys)
    res = calc_derivs_surface_slow(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv,
    )
    res = res.reshape((len(xs), len(ys)) + res.shape[1:])
    if plot_normals:
        normals = calc_normals_surface_slow(
            eval_points[:, 0],
            eval_points[:, 1],
            degree_x,
            degree_y,
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
        )
        normals = normals.reshape((len(xs), len(ys)) + normals.shape[1:])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
        label='control_points',
    )
    ax.plot_wireframe(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
    )
    ax.plot_surface(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        cmap='plasma',
        alpha=0.3,
    )
    ax.quiver(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        res[:, :, 1, 0, 0].detach().cpu().numpy(),
        res[:, :, 1, 0, 1].detach().cpu().numpy(),
        res[:, :, 1, 0, 2].detach().cpu().numpy(),
        length=0.05,
        alpha=0.8,
        label='dS/dx',
    )
    ax.quiver(
        res[:, :, 0, 0, 0].detach().cpu().numpy(),
        res[:, :, 0, 0, 1].detach().cpu().numpy(),
        res[:, :, 0, 0, 2].detach().cpu().numpy(),
        res[:, :, 0, 1, 0].detach().cpu().numpy(),
        res[:, :, 0, 1, 1].detach().cpu().numpy(),
        res[:, :, 0, 1, 2].detach().cpu().numpy(),
        length=0.05,
        color='red',
        alpha=0.8,
        label='dS/dy',
    )
    if plot_normals:
        ax.quiver(
            res[:, :, 0, 0, 0].detach().cpu().numpy(),
            res[:, :, 0, 0, 1].detach().cpu().numpy(),
            res[:, :, 0, 0, 2].detach().cpu().numpy(),
            normals[:, :, 0].detach().cpu().numpy(),
            normals[:, :, 1].detach().cpu().numpy(),
            normals[:, :, 2].detach().cpu().numpy(),
            length=0.05,
            color='green',
            alpha=0.8,
            label='normals',
        )
    ax.legend()
    if show_plot:
        plt.show()
    return fig, ax


def plot_surface_normals(
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        step_granularity_x=0.02,
        step_granularity_y=0.02,
        show_plot=True,
):
    device = control_points.device
    xs = th.arange(0, 1, step_granularity_x, device=device)
    ys = th.arange(0, 1, step_granularity_y, device=device)
    xs = th.hstack([xs, th.tensor(1 - EPS, device=device)])
    ys = th.hstack([ys, th.tensor(1 - EPS, device=device)])

    eval_points = th.cartesian_prod(xs, ys)
    res = evaluate_nurbs_surface_flex(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )
    res = res.reshape((len(xs), len(ys)) + res.shape[1:])
    normals = calc_normals_surface(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )
    normals = normals.reshape((len(xs), len(ys)) + normals.shape[1:])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
        label='control_points',
    )
    ax.plot_wireframe(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
    )
    ax.plot_surface(
        res[:, :, 0].detach().cpu().numpy(),
        res[:, :, 1].detach().cpu().numpy(),
        res[:, :, 2].detach().cpu().numpy(),
        cmap='plasma',
        alpha=0.3,
    )
    ax.quiver(
        res[:, :, 0].detach().cpu().numpy(),
        res[:, :, 1].detach().cpu().numpy(),
        res[:, :, 2].detach().cpu().numpy(),
        normals[:, :, 0].detach().cpu().numpy(),
        normals[:, :, 1].detach().cpu().numpy(),
        normals[:, :, 2].detach().cpu().numpy(),
        length=0.05,
        color='green',
        alpha=0.8,
        label='normals',
    )
    ax.legend()
    if show_plot:
        plt.show()
    return fig, ax


def plot_surface_normals_slow(
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        step_granularity_x=0.02,
        step_granularity_y=0.02,
        show_plot=True,
):
    device = control_points.device
    xs = th.arange(0, 1, step_granularity_x, device=device)
    ys = th.arange(0, 1, step_granularity_y, device=device)
    xs = th.hstack([xs, th.tensor(1 - EPS, device=device)])
    ys = th.hstack([ys, th.tensor(1 - EPS, device=device)])

    eval_points = th.cartesian_prod(xs, ys)
    res, normals = calc_normals_and_surface_slow(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )
    res = res.reshape((len(xs), len(ys)) + res.shape[1:])
    normals = normals.reshape((len(xs), len(ys)) + normals.shape[1:])

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
        label='control_points',
    )
    ax.plot_wireframe(
        control_points[:, :, 0].detach().cpu().numpy(),
        control_points[:, :, 1].detach().cpu().numpy(),
        control_points[:, :, 2].detach().cpu().numpy(),
        color='black',
        alpha=0.1,
    )
    ax.plot_surface(
        res[:, :, 0].detach().cpu().numpy(),
        res[:, :, 1].detach().cpu().numpy(),
        res[:, :, 2].detach().cpu().numpy(),
        cmap='plasma',
        alpha=0.3,
    )
    ax.quiver(
        res[:, :, 0].detach().cpu().numpy(),
        res[:, :, 1].detach().cpu().numpy(),
        res[:, :, 2].detach().cpu().numpy(),
        normals[:, :, 0].detach().cpu().numpy(),
        normals[:, :, 1].detach().cpu().numpy(),
        normals[:, :, 2].detach().cpu().numpy(),
        length=0.05,
        color='green',
        alpha=0.8,
        label='normals',
    )
    ax.legend()
    if show_plot:
        plt.show()
    return fig, ax


def get_inversion_start_values(
        world_points,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        num_samples,
        norm_p=2,
):
    device = control_points.device

    start_spans_x = th.arange(degree_x, len(knots_x) - degree_x, device=device)
    start_spans_y = th.arange(degree_y, len(knots_y) - degree_y, device=device)

    evaluation_points_x = th.hstack([
        th.linspace(
            knots_x[span_x],
            knots_x[span_x + 1] - EPS,
            num_samples,
            device=device,
        )
        for span_x in start_spans_x[:-1]
    ])
    evaluation_points_y = th.hstack([
        th.linspace(
            knots_y[span_y],
            knots_y[span_y + 1] - EPS,
            num_samples,
            device=device,
        )
        for span_y in start_spans_y[:-1]
    ])
    evaluation_points = th.cartesian_prod(
        evaluation_points_x, evaluation_points_y)
    del start_spans_x
    del start_spans_y
    del evaluation_points_x
    del evaluation_points_y

    surface_points = evaluate_nurbs_surface_flex(
        evaluation_points[:, 0],
        evaluation_points[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
    )

    distances = th.linalg.norm(
        surface_points.unsqueeze(0) - world_points.unsqueeze(1),
        ord=norm_p,
        dim=-1,
    )
    min_distances, argmin_distances = distances.min(1)
    return evaluation_points[argmin_distances], min_distances


def batch_dot(x, y):
    return (x * y).sum(-1).unsqueeze(-1)


def invert_points(
        world_points,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        num_samples=8,
        norm_p=2,
        distance_tolerance=1e-5,
        cosine_tolerance=EPS,
):
    argmin_distances, min_distances = get_inversion_start_values(
        world_points,
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        num_samples,
        norm_p=norm_p,
    )

    point_min = 0
    point_max = 1 - EPS

    derivs = calc_derivs_surface(
        argmin_distances[:, 0],
        argmin_distances[:, 1],
        degree_x,
        degree_y,
        control_points,
        control_point_weights,
        knots_x,
        knots_y,
        nth_deriv=2,
    )

    surface_points = derivs[:, 0, 0]

    point_difference = surface_points - world_points

    while True:
        Su = derivs[:, 1, 0]
        Sv = derivs[:, 0, 1]

        points_coincide = (min_distances <= distance_tolerance).all()
        have_zero_cosine = (
            (
                (
                    th.linalg.norm(
                        batch_dot(Su, point_difference),
                        ord=norm_p,
                        dim=-1,
                    )
                    / (th.linalg.norm(Su, ord=norm_p, dim=-1) * min_distances)
                ) <= cosine_tolerance
            ).all()
            or (
                (
                    th.linalg.norm(
                        batch_dot(Sv, point_difference),
                        ord=norm_p,
                        dim=-1,
                    )
                    / (th.linalg.norm(Sv, ord=norm_p, dim=-1) * min_distances)
                ) <= cosine_tolerance
            ).all()
        )
        if points_coincide and have_zero_cosine:
            break

        both_dir_dot = batch_dot(point_difference, derivs[:, 1, 1])

        J = th.stack([
            th.hstack([
                (
                    th.linalg.norm(Su, ord=norm_p, dim=-1).pow(2).unsqueeze(-1)
                    + batch_dot(point_difference, derivs[:, 2, 0])
                ),
                batch_dot(Su, Sv) + both_dir_dot,
            ]),
            th.hstack([
                batch_dot(Su, Sv) + both_dir_dot,
                (
                    th.linalg.norm(Su, ord=norm_p, dim=-1).pow(2).unsqueeze(-1)
                    + batch_dot(point_difference, derivs[:, 0, 2])
                ),
            ]),
        ], dim=1)
        kappa = -th.hstack([
            batch_dot(point_difference, Su),
            batch_dot(point_difference, Sv),
        ])

        delta = th.linalg.solve(J, kappa)

        prev_argmin_distances = argmin_distances
        argmin_distances = delta + prev_argmin_distances

        argmin_distances = argmin_distances.clamp(point_min, point_max)

        # TODO We always assume non-closed surfaces.
        # argmin_distance_x = argmin_distance[:, 0]
        # argmin_distance_y = argmin_distance[:, 1]

        # argmin_distance_x = argmin_distance_x.clamp(point_min, point_max)
        # argmin_distance_y = argmin_distance_y.clamp(point_min, point_max)

        # argmin_distance = th.stack([
        #     argmin_distance_x,
        #     argmin_distance_y,
        # ], dim=-1)

        derivs = calc_derivs_surface(
            argmin_distances[:, 0],
            argmin_distances[:, 1],
            degree_x,
            degree_y,
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
            nth_deriv=2,
        )

        surface_point = derivs[:, 0, 0]

        point_difference = surface_point - world_points
        prev_min_distances = min_distances
        min_distances = th.linalg.norm(
            point_difference,
            ord=norm_p,
            dim=-1,
        )

        error_indices = min_distances > prev_min_distances
        argmin_distances = th.where(
            error_indices.unsqueeze(-1),
            prev_argmin_distances,
            argmin_distances,
        )
        min_distances = th.where(
            error_indices, prev_min_distances, min_distances)
        if error_indices.all():
            # FIXME why does this happen?
            argmin_distances = prev_argmin_distances
            min_distances = prev_min_distances
            break

        have_insignificant_change = (
            th.linalg.norm(
                (
                    (
                        argmin_distances[:, 0]
                        - prev_argmin_distances[:, 0]
                    ).unsqueeze(-1) * Su
                    + (
                        argmin_distances[:, 1]
                        - prev_argmin_distances[:, 1]
                    ).unsqueeze(-1) * Sv
                ),
                ord=norm_p,
                dim=-1,
            ) <= distance_tolerance
        ).all()
        if have_insignificant_change:
            break
    return argmin_distances, min_distances


# def rational_basis_surface_flex(
#         evaluation_point_x,
#         evaluation_point_y,
#         degree_x,
#         degree_y,
#         control_point_weights,
#         knots_x,
#         knots_y,
# ):
#     device = control_point_weights.device

#     control_point_indices_x = th.arange(
#         control_point_weights.shape[0], device=device)
#     basis_values_x = th.tensor([
#         get_basis(evaluation_point_x, control_point_index, degree_x, knots_x)
#         for control_point_index in control_point_indices_x
#     ], dtype=th.float32, device=device)

#     control_point_indices_y = th.arange(
#         control_point_weights.shape[1], device=device)
#     basis_values_y = th.tensor([
#         get_basis(evaluation_point_y, control_point_index, degree_y, knots_y)
#         for control_point_index in control_point_indices_y
#     ], dtype=th.float32, device=device)

#     projected_weights = (
#         basis_values_x
#         * basis_values_y
#         * control_point_weights
#     )
#     return (
#         projected_weights
#         / th.sum(projected_weights)
#     )


# def evaluate_nurbs_surface_flex(
#     evaluation_point_x,
#     evaluation_point_y,
#     degree_x,
#     degree_y,
#     control_points,
#     control_point_weights,
#     knots_x,
#     knots_y,
# ):
#     assert control_points.ndim == 2, (
#         "please use another evaluation function for "
#         "this NURBS' dimensionality"
#     )
#     assert (control_point_weights > 0).all(), \
#         'control point weights must be greater than zero'
#     assert (knots_x[:degree_x + 1] == 0).all(), \
#         f'first {degree_x + 1} knots must be zero'
#     assert (knots_x[control_points.shape[0]] == 1).all(), \
#         f'last {degree_x + 1} knots must be one'
#     assert (knots_x.sort().values == knots_x).all(), \
#         'knots must be ordered monotonically increasing in value'
#     assert (knots_y[:degree_y + 1] == 0).all(), \
#         f'first {degree_y + 1} knots must be zero'
#     assert (knots_y[control_points.shape[1]] == 1).all(), \
#         f'last {degree_y + 1} knots must be one'
#     assert (knots_y.sort().values == knots_y).all(), \
#         'knots must be ordered monotonically increasing in value'

#     rational_basis_values = rational_basis_surface_flex(
#             evaluation_point_x,
#             evaluation_point_y,
#             degree_x,
#             degree_y,
#             control_point_weights,
#             knots_x,
#             knots_y,
#     )
#     return th.sum(rational_basis_values * control_points)

    # device = control_point_weights_x.device

    # control_point_indices_x = th.arange(
    #     len(control_point_weights_x), device=device)
    # basis_values_x = th.tensor([
    #     get_basis(evaluation_point_x, control_point_index, degree_x, knots_x)
    #     for control_point_index in control_point_indices_x
    # ], dtype=th.float32, device=device)

    # control_point_indices_y = th.arange(
    #     len(control_point_weights_y), device=device)
    # basis_values_y = th.tensor([
    #     get_basis(evaluation_point_y, control_point_index, degree_y, knots_y)
    #     for control_point_index in control_point_indices_y
    # ], dtype=th.float32, device=device)

    # control_point_weights = th.outer(
    #     control_point_weights_x, control_point_weights_y)
    # projected_weights = (
    #     basis_values_x
    #     * basis_values_y
    #     * control_point_weights
    # )
    # return (
    #     projected_weights
    #     / th.sum(projected_weights)
    # )


# def calc_normal_surface():
#     x_derivative = th.autograd.functional.jacobian()
#     cross_prod =
#     return cross_prod / th.linalg.norm(cross_prod)


# def rational_basis_surface(
#         evaluation_point,
#         degree,
#         control_point_weights,
#         knots,
# ):
#     device = control_point_weights.device

#     control_point_indices = th.arange(
#         len(control_point_weights),
#         device=device,
#     )
#     basis_values = th.tensor([
#         get_basis(evaluation_point, control_point_index, degree, knots)
#         for control_point_index in control_point_indices
#     ], dtype=th.float32, device=device)
#     projected_weights = basis_values * control_point_weights
#     pass


# def evaluate_nurbs_surface(
#     evaluation_point,
#     degree,
#     control_points,
#     control_point_weights,
#     knots,
# ):
#     projected_weights = basis_values * control_point_weights
#     return (
#         th.sum(projected_weights * control_points)
#         / th.sum(projected_weights)
#     )


class NURBSSurface:
    def __init__(
            self,
            degree_x,
            degree_y,
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
    ):
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.control_points = control_points
        self.control_point_weights = control_point_weights
        self.knots_x = knots_x
        self.knots_y = knots_y

    @classmethod
    def create_empty(
            cls,
            degree_x,
            degree_y,
            num_control_points_x,
            num_control_points_y,
            device,
    ):
        (
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
        ) = setup_nurbs_surface(
            degree_x,
            degree_y,
            num_control_points_x,
            num_control_points_y,
            device,
        )
        return cls(
            degree_x,
            degree_y,
            control_points,
            control_point_weights,
            knots_x,
            knots_y,
        )

    @classmethod
    def create_example(cls, device=th.device('cpu')):
        degree = 3
        num_ctrl = 6
        surf = cls.create_empty(degree, degree, num_ctrl, num_ctrl, device)

        y_inds, x_inds = th.meshgrid(
            th.linspace(0, 1, num_ctrl),
            th.linspace(0, 1, num_ctrl),
        )

        surf.control_points[:, :, 0] = y_inds
        surf.control_points[:, :, 1] = x_inds

        first_circle_height = 1/3
        surf.control_points[0, :, 2] = first_circle_height
        surf.control_points[:, 0, 2] = first_circle_height
        surf.control_points[-1, :, 2] = first_circle_height
        surf.control_points[:, -1, 2] = first_circle_height

        second_circle_height = 0
        surf.control_points[1, 1:-1, 2] = second_circle_height
        surf.control_points[1:-1, 1, 2] = second_circle_height
        surf.control_points[-2, 1:-1, 2] = second_circle_height
        surf.control_points[1:-1, -2, 2] = second_circle_height

        third_circle_height = 1
        surf.control_points[2, 2:-2, 2] = third_circle_height
        surf.control_points[2:-2, 2, 2] = third_circle_height
        surf.control_points[-3, 2:-2, 2] = third_circle_height
        surf.control_points[2:-2, -3, 2] = third_circle_height

        surf.control_point_weights[:] = 1

        surf.knots_x[degree:-degree] = th.linspace(
            0, 1, len(surf.knots_x[degree:-degree]))
        surf.knots_y[degree:-degree] = th.linspace(
            0, 1, len(surf.knots_y[degree:-degree]))
        # surf.knots_x[:] = 1
        # surf.knots_y[:] = 1
        # surf.knots_x[degree + 1] = 0
        # surf.knots_x[-degree - 2] = 1
        # surf.knots_y[degree + 1] = 0
        # surf.knots_y[-degree - 2] = 1
        return surf

    def evaluate(self, evaluation_point_x, evaluation_point_y):
        return evaluate_nurbs_surface_flex(
            evaluation_point_x,
            evaluation_point_y,
            self.degree_x,
            self.degree_y,
            self.control_points,
            self.control_point_weights,
            self.knots_x,
            self.knots_y,
        )

    def calc_bspline_derivs(
            self,
            evaluation_point_x,
            evaluation_point_y,
            nth_deriv=1,
    ):
        return calc_bspline_derivs_surface(
                evaluation_point_x,
                evaluation_point_y,
                self.degree_x,
                self.degree_y,
                self.control_points,
                self.knots_x,
                self.knots_y,
                nth_deriv,
        )

    def calc_derivs(
            self,
            evaluation_point_x,
            evaluation_point_y,
            nth_deriv=1,
    ):
        return calc_derivs_surface(
            evaluation_point_x,
            evaluation_point_y,
            self.degree_x,
            self.degree_y,
            self.control_points,
            self.control_point_weights,
            self.knots_x,
            self.knots_y,
            nth_deriv,
        )

    def plot(
            self,
            step_granularity_x=0.02,
            step_granularity_y=0.02,
            show_plot=True,
    ):
        return plot_surface(
            self.degree_x,
            self.degree_y,
            self.control_points,
            self.control_point_weights,
            self.knots_x,
            self.knots_y,
            step_granularity_x,
            step_granularity_y,
            show_plot,
        )

    def plot_derivs(
            self,
            step_granularity_x=0.1,
            step_granularity_y=0.1,
            nth_deriv=1,
            show_plot=True,
            plot_normals=None,
    ):
        if plot_normals is None:
            plot_normals = nth_deriv == 1
        return plot_surface_derivs(
            self.degree_x,
            self.degree_y,
            self.control_points,
            self.control_point_weights,
            self.knots_x,
            self.knots_y,
            step_granularity_x,
            step_granularity_y,
            nth_deriv,
            show_plot,
            plot_normals,
        )
