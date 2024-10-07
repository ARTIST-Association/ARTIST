import torch


class NURBSSurface(torch.nn.Module):
    """
    Implement differentiable NURBS for the heliostat surface.

    Attributes
    ----------
    degree_e : int
        The spline degree in east direction.
    degree_n : int
        The spline degree in north direction.
    evaluation_points_e : torch.Tensor
        The evaluation points in east direction.
    evaluation_points_n : torch.Tensor
        The evaluation points in north direction.
    control_points : torch.Tensor
        The control_points.
    knot_vector_e : torch.Tensor
        The knot vector in east direction.
    knot_vector_n : torch.Tensor
        The knot vector in north direction.

    Methods
    -------
    calculate_knots()
        Calculate the knot vectors in east and north direction.
    find_span()
         Determine the knot span index for given evaluation points.
    basis_function_and_derivatives()
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.
    calculate_surface_points_and_normals()
        Calculate the surface points and normals of the NURBS surface.
    """

    def __init__(
        self,
        degree_e: int,
        degree_n: int,
        evaluation_points_e: torch.Tensor,
        evaluation_points_n: torch.Tensor,
        control_points: torch.Tensor,
    ) -> None:
        """
        Initialize a NURBS surface.

        NURBS stands for Non-Uniform Rational B-Splines and allow for an efficient and precise reconstruction
        of the imperfect heliostat surfaces in the digital twin. This implementation of the NURBS is
        differentiable. The NURBS surfaces require a degree in two directions, evaluation points, and control
        points. These parameters are used to create the NURBS surface. For more details, see the NURBS tutorial.

        Parameters
        ----------
        degree_e : int
            The spline degree in east direction.
        degree_n : int
            The spline degree in north direction.
        evaluation_points_e : torch.Tensor
            The evaluation points in east direction.
        evaluation_points_n : torch.Tensor
            The evaluation points in north direction.
        control_points : torch.Tensor
            The control_points.
        """
        super().__init__()
        self.degree_e = degree_e
        self.degree_n = degree_n
        self.evaluation_points_e = evaluation_points_e
        self.evaluation_points_n = evaluation_points_n
        self.control_points = control_points
        self.knot_vector_e, self.knot_vector_n = self.calculate_knots()

    def calculate_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the knot vectors in the east and north direction.

        For our application, only uniform knot vectors are required.
        The knots range from zero to one and are distributed uniformly.
        The first knot (0) and the last knot (1) have full multiplicity,
        i.e., they are repeated as often as specified by the degree.
        This means the NURBS start and end in a control point.

        Returns
        -------
        torch.Tensor
            The knots in east direction.
        torch.Tensor
            The knots in north direction.
        """
        num_control_points_e = self.control_points.shape[0]
        num_control_points_n = self.control_points.shape[1]

        knots_e = torch.zeros(num_control_points_e + self.degree_e + 1)
        num_knot_vals = len(knots_e[self.degree_e : -self.degree_e])
        knot_vals = torch.linspace(0, 1, num_knot_vals)
        knots_e[: self.degree_e] = 0
        knots_e[self.degree_e : -self.degree_e] = knot_vals
        knots_e[-self.degree_e :] = 1

        knots_n = torch.zeros(num_control_points_n + self.degree_n + 1)
        num_knot_vals = len(knots_n[self.degree_n : -self.degree_n])
        knot_vals = torch.linspace(0, 1, num_knot_vals)
        knots_n[: self.degree_n] = 0
        knots_n[self.degree_n : -self.degree_n] = knot_vals
        knots_n[-self.degree_n :] = 1

        return knots_e, knots_n

    @staticmethod
    def find_span(
        degree: int,
        evaluation_points: torch.Tensor,
        knot_vector: torch.Tensor,
        control_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Determine the knot span index for given evaluation points.

        To generate NURBS, the basis functions must be evaluated. However, some basis functions may be zero. To improve
        computational efficiency, basis functions that are zero are not computed. Therefore, the knot span in which the
        evaluation point lies is first computed using this function.
        See `The NURBS Book` p. 68 for reference.

        Parameters
        ----------
        degree : int
            The degree of the NURBS surface in a single direction.
        evaluation_points : torch.Tensor
            The evaluation points.
        knot_vector : torch.Tensor
            The knot vector for the NURBS surface in a single direction.
        control_points : torch.Tensor
            The control points.

        Returns
        -------
        torch.Tensor
            The knot span index.
        """
        n = control_points.shape[1] - 1
        span_indices = torch.zeros(len(evaluation_points), dtype=torch.int64)
        for i, evaluation_point in enumerate(evaluation_points):
            if torch.isclose(evaluation_point, knot_vector[n], atol=1e-5, rtol=1e-5):
                span_indices[i] = n
                continue
            low = degree
            high = control_points.shape[1]
            mid = (low + high) // 2
            while (
                evaluation_point < knot_vector[mid]
                or evaluation_point >= knot_vector[mid + 1]
            ):
                if evaluation_point < knot_vector[mid]:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            span_indices[i] = mid

        return span_indices

    @staticmethod
    def basis_function_and_derivatives(
        evaluation_points: torch.Tensor,
        knot_vector: torch.Tensor,
        span: torch.Tensor,
        degree: int,
        nth_derivative: int = 1,
    ) -> torch.Tensor:
        """
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.

        See `The NURBS Book` p. 72 for reference.

        Parameters
        ----------
        evaluation_points : torch.Tensor
            The evaluation points.
        knot_vector : torch.Tensor
            Contains all the knots of the NURBS model.
        span : torch.Tensor
            The span indices.
        degree : int
            The degree of the NURBS surface in one direction.
        nth_derivative : int
            Specifies how many derivatives are calculated (default: 1).

        Returns
        -------
        torch.Tensor
            The derivatives of the basis function.
        """
        num_evaluation_points = len(evaluation_points)

        # Introduce `ndu` to store the basis functions (called "n" in The NURBS book) and the knot differences (du).
        ndu = [
            [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
            for _ in range(degree + 1)
        ]
        ndu[0][0] = torch.ones_like(ndu[0][0])
        left = [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
        right = [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
        for j in range(1, degree + 1):
            left[j] = evaluation_points - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - evaluation_points
            saved = torch.zeros(num_evaluation_points)
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                # Introduce `tmp` to temporarily store result.
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        derivatives = [
            [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
            for _ in range(nth_derivative + 1)
        ]
        for j in range(degree + 1):
            derivatives[0][j] = ndu[j][degree]
        # `a` stores (in alternating fashion) the two most recently computed rows a_k,j and a_k-1,j.
        a = [
            [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
            for _ in range(2)
        ]
        for r in range(degree + 1):
            s1 = 0
            s2 = 1
            a[0][0] = torch.ones_like(a[0][0])
            for k in range(1, nth_derivative + 1):
                d = torch.zeros(num_evaluation_points)
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
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                    d += a[s2][j] * ndu[rk + j][pk]
                if r <= pk:
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                    d += a[s2][k] * ndu[r][pk]
                derivatives[k][r] = d
                j = s1
                s1 = s2
                s2 = j

        r = degree
        for k in range(1, nth_derivative + 1):
            for j in range(degree + 1):
                derivatives[k][j] *= r
            r *= degree - k
        return derivatives

    def calculate_surface_points_and_normals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the surface points and normals of the NURBS surface.

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface normals.
        """
        nth_derivative = 1

        # Find span indices x direction (based on A2.1, p. 68).
        span_indices_e = self.find_span(
            self.degree_e,
            self.evaluation_points_e,
            self.knot_vector_e,
            self.control_points,
        )

        # Find span indices y direction (based on A2.1, p. 68).
        span_indices_n = self.find_span(
            self.degree_n,
            self.evaluation_points_n,
            self.knot_vector_n,
            self.control_points,
        )

        control_point_weights = torch.ones(
            (self.control_points.shape[0], self.control_points.shape[1]) + (1,)
        )
        control_points = torch.cat([self.control_points, control_point_weights], dim=-1)

        derivatives = torch.zeros(
            len(self.evaluation_points_e),
            nth_derivative + 1,
            nth_derivative + 1,
            control_points.shape[-1],
        )

        # Find minimum of `nth_derivative` and degree, will be used to specify how many partial derivatives will be
        # computed.
        de = min(nth_derivative, self.degree_e)
        for k in range(self.degree_e + 1, nth_derivative + 1):
            for t in range(nth_derivative - k + 1):
                derivatives[:, k, t] = 0
        dn = min(nth_derivative, self.degree_n)
        for t in range(self.degree_n + 1, nth_derivative + 1):
            for k in range(nth_derivative - t + 1):
                derivatives[:, k, t] = 0

        # Find derivatives of basis functions (based on A2.3, p. 72).
        basis_values_derivatives_e = self.basis_function_and_derivatives(
            self.evaluation_points_e,
            self.knot_vector_e,
            span_indices_e,
            self.degree_e,
            de,
        )
        basis_values_derivatives_n = self.basis_function_and_derivatives(
            self.evaluation_points_n,
            self.knot_vector_n,
            span_indices_n,
            self.degree_n,
            dn,
        )

        # Find surface points and normals (based on A3.6, p. 111).
        # `temp` stores the vector/matrix product of the basis value derivatives and the control points.
        temp = [
            torch.zeros((len(self.evaluation_points_e), control_points.shape[-1]))
            for _ in range(self.degree_n + 1)
        ]
        for k in range(de + 1):
            for s in range(self.degree_n + 1):
                temp[s] = torch.zeros_like(temp[s])
                for r in range(self.degree_e + 1):
                    temp[s] += (
                        basis_values_derivatives_e[k][r].unsqueeze(-1)
                        * control_points[
                            span_indices_e - self.degree_e + r,
                            span_indices_n - self.degree_n + s,
                        ]
                    )
            dd = min(nth_derivative - k, dn)
            for t in range(dd + 1):
                derivatives[:, k, t] = 0
                for s in range(self.degree_n + 1):
                    derivatives[:, k, t] += (
                        basis_values_derivatives_n[t][s].unsqueeze(-1) * temp[s]
                    )

        normals = torch.linalg.cross(derivatives[:, 1, 0, :3], derivatives[:, 0, 1, :3])
        normals = torch.nn.functional.normalize(normals)

        normals = torch.cat((normals, torch.zeros(normals.shape[0], 1)), dim=1)

        return derivatives[:, 0, 0], normals
