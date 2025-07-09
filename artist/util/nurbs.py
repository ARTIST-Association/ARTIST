import torch

from artist.util.environment_setup import get_device


class NURBSSurface(torch.nn.Module):
    """
    Implement differentiable NURBS for the heliostat surface.

    Attributes
    ----------
    degrees : torch.Tensor
        The spline degrees.
    control_points : torch.Tensor
        The control points.
    knot_vector_e : torch.Tensor
        The knot vector in east direction.
    knot_vector_n : torch.Tensor
        The knot vector in north direction.

    Methods
    -------
    calculate_knot_vector()
        Calculate the knot vector in one dimension.
    find_span()
        Determine the knot span index in one dimension.
    basis_function_and_derivatives()
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.
    calculate_surface_points_and_normals()
        Calculate the surface points and normals of the NURBS surface.
    forward()
        Specify the forward operation of the actuator, i.e. caluclate the surface points and normals.
    """

    def __init__(
        self,
        degrees: torch.Tensor,
        control_points: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a NURBS surface.

        NURBS stands for Non-Uniform Rational B-Spline. NURBS allow for an efficient and precise reconstruction
        of the imperfect heliostat surfaces in the digital twin. This implementation of the NURBS is
        differentiable. The NURBS surfaces require a degree in two directions, evaluation points, and control
        points. These parameters are used to create the NURBS surface. For more details, see the NURBS tutorial.

        Parameters
        ----------
        degrees : torch.tensor
            The spline degrees in east and north directions.
        control_points : torch.Tensor
            The control points.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.degrees = degrees
        self.control_points = torch.nn.Parameter(control_points)
        self.knot_vector_e = self.calculate_knot_vector(dimension=0, device=device)
        self.knot_vector_n = self.calculate_knot_vector(dimension=1, device=device)

    def calculate_knot_vector(
        self,
        dimension: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Calculate the knot vector in one dimension.

        For our application, only uniform knot vectors are required.
        The knots range from zero to one and are distributed uniformly.
        The first knot (0) and the last knot (1) have full multiplicity,
        i.e., they are repeated as often as specified by the degree.
        This means the NURBS start and end in a control point.

        Parameters
        ----------
        dimension : int
            The NURBS dimension.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The knot vector.
        """
        device = get_device(device=device)

        knot_vector = torch.zeros(
            self.control_points.shape[dimension] + self.degrees[dimension] + 1,
            device=device,
        )
        number_of_knot_values = len(
            knot_vector[self.degrees[dimension] : -self.degrees[dimension]]
        )
        knot_vector[: self.degrees[dimension]] = 0
        knot_vector[self.degrees[dimension] : -self.degrees[dimension]] = (
            torch.linspace(0, 1, number_of_knot_values, device=device)
        )
        knot_vector[-self.degrees[dimension] :] = 1

        return knot_vector

    def find_span(
        self,
        dimension: int,
        evaluation_points: torch.Tensor,
        knot_vector: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Determine the knot span index in one dimension.

        To generate NURBS, the basis functions must be evaluated. However, some basis functions may be zero. To improve
        computational efficiency, basis functions that are zero are not computed. Therefore, the knot span in which the
        evaluation point lies is first computed using this function.
        See `The NURBS Book` p. 68 for reference.
        If the knot vector is uniform, the span indices can be computed more efficiently.

        Parameters
        ----------
        dimension : int
            The NURBS dimension.
        evaluation_points : torch.Tensor
            The evaluation points.
        knot_vector : torch.Tensor
            The knot vector for the NURBS surface in a single direction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The knot span index.
        """
        device = get_device(device=device)

        knot_vector_is_uniform = torch.all(
            (
                torch.diff(
                    knot_vector[self.degrees[dimension] : -self.degrees[dimension]]
                )
                - torch.diff(
                    knot_vector[self.degrees[dimension] : -self.degrees[dimension]]
                )[0]
            )
            < 1e-5
        )
        if knot_vector_is_uniform:
            unique_knots = torch.unique(knot_vector)

            min_value = unique_knots[0]
            max_value = unique_knots[-1]

            scaled_points = (evaluation_points[:, dimension] - min_value) / (
                max_value - min_value
            )

            span_indices = (
                torch.floor(scaled_points * (len(unique_knots) - 1)).long()
                + self.degrees[dimension]
            )

        else:
            n = self.control_points.shape[1] - 1
            span_indices = torch.zeros(
                len(evaluation_points[:, dimension]),
                dtype=torch.int64,
                device=device,
            )
            for i, evaluation_point in enumerate(evaluation_points[:, dimension]):
                if torch.isclose(
                    evaluation_point, knot_vector[n], atol=1e-5, rtol=1e-5
                ):
                    span_indices[i] = n
                    continue
                low = self.degrees[dimension]
                high = self.control_points.shape[1]
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

    def basis_function_and_derivatives(
        self,
        dimension: int,
        evaluation_points: torch.Tensor,
        knot_vector: torch.Tensor,
        span: torch.Tensor,
        nth_derivative: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.

        See `The NURBS Book` p. 72 for reference.

        Parameters
        ----------
        dimension : int
            The NURBS dimension.
        evaluation_points : torch.Tensor
            The evaluation points.
        knot_vector : torch.Tensor
            Contains all the knots of the NURBS model.
        span : torch.Tensor
            The span indices.
        nth_derivative : int
            Specifies how many derivatives are calculated (default is 1).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The derivatives of the basis function.
        """
        device = get_device(device=device)

        num_evaluation_points = len(evaluation_points[:, dimension])

        # Introduce `ndu` to store the basis functions (called "n" in The NURBS book) and the knot differences (du).
        ndu = [
            [
                torch.zeros(num_evaluation_points, device=device)
                for _ in range(self.degrees[dimension] + 1)
            ]
            for _ in range(self.degrees[dimension] + 1)
        ]
        ndu[0][0] = torch.ones_like(ndu[0][0], device=device)
        left = [
            torch.zeros(num_evaluation_points, device=device)
            for _ in range(self.degrees[dimension] + 1)
        ]
        right = [
            torch.zeros(num_evaluation_points, device=device)
            for _ in range(self.degrees[dimension] + 1)
        ]
        for j in range(1, self.degrees[dimension] + 1):
            left[j] = evaluation_points[:, dimension] - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - evaluation_points[:, dimension]
            saved = torch.zeros(num_evaluation_points, device=device)
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                # Introduce `tmp` to temporarily store result.
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        derivatives = [
            [
                torch.zeros(num_evaluation_points, device=device)
                for _ in range(self.degrees[dimension] + 1)
            ]
            for _ in range(nth_derivative + 1)
        ]
        for j in range(self.degrees[dimension] + 1):
            derivatives[0][j] = ndu[j][self.degrees[dimension]]
        # `a` stores (in alternating fashion) the two most recently computed rows a_k,j and a_k-1,j.
        a = [
            [
                torch.zeros(num_evaluation_points, device=device)
                for _ in range(self.degrees[dimension] + 1)
            ]
            for _ in range(2)
        ]
        for r in range(self.degrees[dimension] + 1):
            s1 = 0
            s2 = 1
            a[0][0] = torch.ones_like(a[0][0], device=device)
            for k in range(1, nth_derivative + 1):
                d = torch.zeros(num_evaluation_points, device=device)
                rk = r - k
                pk = self.degrees[dimension] - k
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
                    j2 = self.degrees[dimension] - r
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

        r = self.degrees[dimension]
        for k in range(1, nth_derivative + 1):
            for j in range(self.degrees[dimension] + 1):
                derivatives[k][j] *= r
            r *= self.degrees[dimension] - k
        return derivatives

    def calculate_surface_points_and_normals(
        self,
        evaluation_points: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the surface points and normals of the NURBS surface.

        Parameters
        ----------
        evaluation_points : torch.Tensor
            The evaluation points.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface normals.
        """
        device = get_device(device=device)

        nth_derivative = 1

        # Find span indices e direction (based on A2.1, p. 68).
        span_indices_e = self.find_span(
            dimension=0,
            evaluation_points=evaluation_points,
            knot_vector=self.knot_vector_e,
            device=device,
        )

        # Find span indices n direction (based on A2.1, p. 68).
        span_indices_n = self.find_span(
            dimension=1,
            evaluation_points=evaluation_points,
            knot_vector=self.knot_vector_n,
            device=device,
        )

        control_point_weights = torch.ones(
            (self.control_points.shape[0], self.control_points.shape[1]) + (1,),
            device=device,
        )
        control_points = torch.cat([self.control_points, control_point_weights], dim=-1)

        derivatives = torch.zeros(
            len(evaluation_points[:, 0]),
            nth_derivative + 1,
            nth_derivative + 1,
            control_points.shape[-1],
            device=device,
        )

        # Find minimum of `nth_derivative` and degree, will be used to specify how many partial derivatives will be
        # computed.
        de = min(nth_derivative, self.degrees[0])
        for k in range(self.degrees[0] + 1, nth_derivative + 1):
            for t in range(nth_derivative - k + 1):
                derivatives[:, k, t] = 0
        dn = min(nth_derivative, self.degrees[1])
        for t in range(self.degrees[1] + 1, nth_derivative + 1):
            for k in range(nth_derivative - t + 1):
                derivatives[:, k, t] = 0

        # Find derivatives of basis functions (based on A2.3, p. 72).
        basis_values_derivatives_e = self.basis_function_and_derivatives(
            dimension=0,
            evaluation_points=evaluation_points,
            knot_vector=self.knot_vector_e,
            span=span_indices_e,
            nth_derivative=de,
            device=device,
        )
        basis_values_derivatives_n = self.basis_function_and_derivatives(
            dimension=1,
            evaluation_points=evaluation_points,
            knot_vector=self.knot_vector_n,
            span=span_indices_n,
            nth_derivative=dn,
            device=device,
        )

        # Find surface points and normals (based on A3.6, p. 111).
        # `temp` stores the vector/matrix product of the basis value derivatives and the control points.
        temp = [
            torch.zeros(
                (len(evaluation_points[:, 0]), control_points.shape[-1]),
                device=device,
            )
            for _ in range(self.degrees[1] + 1)
        ]
        for k in range(de + 1):
            for s in range(self.degrees[1] + 1):
                temp[s] = torch.zeros_like(temp[s], device=device)
                for r in range(self.degrees[0] + 1):
                    temp[s] += (
                        basis_values_derivatives_e[k][r].unsqueeze(-1)
                        * control_points[
                            span_indices_e - self.degrees[0] + r,
                            span_indices_n - self.degrees[1] + s,
                        ]
                    )
            dd = min(nth_derivative - k, dn)
            for t in range(dd + 1):
                derivatives[:, k, t] = 0
                for s in range(self.degrees[1] + 1):
                    derivatives[:, k, t] += (
                        basis_values_derivatives_n[t][s].unsqueeze(-1) * temp[s]
                    )

        normals = torch.linalg.cross(derivatives[:, 1, 0, :3], derivatives[:, 0, 1, :3])
        normals = torch.nn.functional.normalize(normals)

        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0], 1, device=device)), dim=1
        )

        return derivatives[:, 0, 0], normals

    def forward(
        self, evaluation_points: torch.Tensor, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Specify the forward operation of the actuator, i.e. caluclate the surface points and normals.

        Parameters
        ----------
        evaluation_points : torch.Tensor
            The evaluation points.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points.
        torch.Tensor
            The surface normals.
        """
        return self.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points, device=device
        )
