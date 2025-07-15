import torch

from artist.util import type_mappings
from artist.util.environment_setup import get_device


class NURBSSurfaces(torch.nn.Module):
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
        uniform: bool = True,
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
        self.uniform = uniform
        self.number_of_surfaces = self.control_points.shape[0]
        self.number_of_facets_per_surface = self.control_points.shape[1]
        # self.number_of_control_points_u = self.control_points.shape[2]
        # self.number_of_control_points_v = self.control_points.shape[3]
        self.knot_vectors_u = self.calculate_knot_vectors(dimension="u", uniform=self.uniform, device=device)
        self.knot_vectors_v = self.calculate_knot_vectors(dimension="v", uniform=self.uniform, device=device)

    def calculate_knot_vectors(
        self,
        dimension: int,
        uniform: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """
        Calculate the knot vectors for all surfaces in one dimension.

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
        torch.Tensor | None
            The knot vectors.
        """
        device = get_device(device=device)

        dimension=type_mappings.nurbs_dimension_mapping[dimension]
        degree = self.degrees[dimension].item()

        if uniform:
            knot_vector = torch.zeros(
                (
                    self.control_points.shape[2 + dimension] + degree + 1
                ),
                device=device,
            )
            number_of_knot_values = knot_vector[degree : -degree].shape[0]
            knot_vector[:degree] = 0
            knot_vector[degree:-degree] = (
                torch.linspace(0, 1, number_of_knot_values, device=device)
            )
            knot_vector[-degree:] = 1

            knot_vectors = knot_vector.unsqueeze(0).repeat(self.number_of_surfaces, self.number_of_facets_per_surface, 1)

        else:
            raise ValueError("Non uniform knots not implemented.")

        return knot_vectors

    def find_spans(
        self,
        dimension: int,
        evaluation_points: torch.Tensor,
        knot_vectors: torch.Tensor,
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

        dimension=type_mappings.nurbs_dimension_mapping[dimension]
        degree = self.degrees[dimension].item()

        if self.uniform:
            unique_knots = torch.unique(knot_vectors, dim=-1)

            span_indices = (
                torch.floor(evaluation_points[:, :, :, dimension] * (unique_knots.shape[2] - 1)).long()
                + self.degrees[dimension]
            )
        
        else:
            number_of_knots = knot_vectors.shape[2] - degree - 1

            valid_spans = []
            for i in range(degree, number_of_knots):
                left = knot_vectors[:, :, i]
                right = knot_vectors[:, :, i+1]
                valid_spans.append((left, right))

            lefts = torch.stack([span[0] for span in valid_spans], dim=2)
            rights = torch.stack([span[1] for span in valid_spans], dim=2)

            in_span = (evaluation_points[:, :, :, 0].unsqueeze(-1) >= lefts.unsqueeze(2)) & (evaluation_points[:, :, :, 0].unsqueeze(-1) < rights.unsqueeze(2))
            is_last_knot = torch.isclose(evaluation_points[:, :, :, 0], knot_vectors[:, :, number_of_knots], atol=1e-5, rtol=1e-5)
            span_indices = torch.argmax(in_span.to(torch.int), dim=-1) + degree
            span_indices = torch.where(is_last_knot, torch.full_like(span_indices, number_of_knots - 1), span_indices)

        return span_indices

    def basis_functions_and_derivatives(
        self,
        dimension: str,
        evaluation_points: torch.Tensor,
        knot_vectors: torch.Tensor,
        spans: torch.Tensor,
        nth_derivative: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.

        See `The NURBS Book` p. 72 for reference.

        Parameters
        ----------
        degree : int
            The NURBS degree.
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

        dimension = type_mappings.nurbs_dimension_mapping[dimension]
        degree = self.degrees[dimension].item()

        evaluation_points = evaluation_points[:, :, :, 0]
        num_evaluation_points = evaluation_points.shape[2]

        # Introduce `ndu` to store the basis functions (called "n" in The NURBS book) and the knot differences (du).
        ndu = [
            [
                torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device)
                for _ in range(degree + 1)
            ]
            for _ in range(degree + 1)
        ]
        ndu[0][0] = torch.ones_like(ndu[0][0], device=device)
        left = [
            torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device) for _ in range(degree + 1)
        ]
        right = [
            torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device) for _ in range(degree + 1)
        ]
        for j in range(1, degree + 1):
            left[j] = evaluation_points - torch.gather(knot_vectors, 2, spans - j + 1)
            right[j] = torch.gather(knot_vectors, 2, spans + j) - evaluation_points
            saved = torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device)
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                # Introduce `tmp` to temporarily store result.
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        derivatives = [
            [
                torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device)
                for _ in range(degree + 1)
            ]
            for _ in range(nth_derivative + 1)
        ]
        for j in range(degree + 1):
            derivatives[0][j] = ndu[j][degree]
        # `a` stores (in alternating fashion) the two most recently computed rows a_k,j and a_k-1,j.
        a = [
            [
                torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device)
                for _ in range(degree + 1)
            ]
            for _ in range(2)
        ]
        for r in range(degree + 1):
            s1 = 0
            s2 = 1
            a[0][0] = torch.ones_like(a[0][0], device=device)
            for k in range(1, nth_derivative + 1):
                d = torch.zeros((self.number_of_surfaces, self.number_of_facets_per_surface, num_evaluation_points), device=device)
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
        
        return [torch.stack(derivatives[k], dim=-1) for k in range(nth_derivative + 1)]

    def batched_gather_control_points(self, control_points: torch.Tensor, index_u: torch.Tensor, index_v: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """
        Gathers control points using batched 2D indices.

        Parameters
        ----------
        control_points : torch.Tensor
            Tensor of shape [B, F, Nu, Nv, D]
        idx_u : torch.Tensor
            Indices in the u-direction [B, F, E]
        idx_v : torch.Tensor
            Indices in the v-direction [B, F, E]

        Returns
        -------
        torch.Tensor
            Gathered control points of shape [B, F, E, D]
        """
        device = get_device(device=device)
        number_of_evaluation_points = index_u.shape[2]

        # Build batch and facet indices
        batch_index = torch.arange(
            self.number_of_surfaces, device=device
        ).view(self.number_of_surfaces, 1, 1).expand(
            self.number_of_surfaces, self.number_of_facets_per_surface, number_of_evaluation_points
        )
        facet_index = torch.arange(
            self.number_of_facets_per_surface, device=device
        ).view(1, self.number_of_facets_per_surface, 1).expand(
            self.number_of_surfaces, self.number_of_facets_per_surface, number_of_evaluation_points
        )

        # Advanced indexing
        gathered = control_points[batch_index, facet_index, index_u, index_v]
        return gathered


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
        span_indices_u = self.find_spans(
            dimension="u",
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_u,
            device=device,
        )

        # Find span indices n direction (based on A2.1, p. 68).
        span_indices_v = self.find_spans(
            dimension="v",
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_v,
            device=device,
        )

        control_point_weights = torch.ones(
            (
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                self.control_points.shape[2], 
                self.control_points.shape[3],
                1
            ),
            device=device,
        )
        control_points = torch.cat([self.control_points, control_point_weights], dim=-1)

        derivatives = torch.zeros(
            self.number_of_surfaces,
            self.number_of_facets_per_surface,
            evaluation_points.shape[2],
            nth_derivative + 1,
            nth_derivative + 1,
            control_points.shape[-1],
            device=device,
        )

        # Find minimum of `nth_derivative` and degree, will be used to specify how many partial derivatives will be
        # computed.
        du = min(nth_derivative, self.degrees[0])
        for k in range(self.degrees[0] + 1, nth_derivative + 1):
            for t in range(nth_derivative - k + 1):
                derivatives[:, :, :, k, t] = 0
        dv = min(nth_derivative, self.degrees[1])
        for t in range(self.degrees[1] + 1, nth_derivative + 1):
            for k in range(nth_derivative - t + 1):
                derivatives[:, :, :, k, t] = 0

        # Find derivatives of basis functions (based on A2.3, p. 72).
        basis_values_derivatives_u = self.basis_functions_and_derivatives(
            dimension="u",
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_u,
            spans=span_indices_u,
            nth_derivative=du,
            device=device,
        )
        basis_values_derivatives_v = self.basis_functions_and_derivatives(
            dimension="v",
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_v,
            spans=span_indices_v,
            nth_derivative=dv,
            device=device,
        )

        # Find surface points and normals (based on A3.6, p. 111).
        # `temp` stores the vector/matrix product of the basis value derivatives and the control points.
        temp = [
            torch.zeros(
                (self.number_of_surfaces, self.number_of_facets_per_surface, evaluation_points.shape[2], control_points.shape[-1]),
                device=device,
            )
            for _ in range(self.degrees[1] + 1)
        ]
        for k in range(du + 1):
            for s in range(self.degrees[1] + 1):
                temp[s] = torch.zeros_like(temp[s], device=device)
                for r in range(self.degrees[0] + 1):
                    bu = basis_values_derivatives_u[k][:, :, :, r].unsqueeze(-1)
                    index_u = span_indices_u - self.degrees[0] + r
                    index_v = span_indices_v - self.degrees[1] + s
                    gathered_control_points = self.batched_gather_control_points(
                        control_points=control_points, 
                        index_u=index_u, 
                        index_v=index_v,
                        device=device
                    )
                    temp[s] += bu * gathered_control_points

            dd = min(nth_derivative - k, dv)
            for t in range(dd + 1):
                derivatives[:, :, :, k, t] = 0
                for s in range(self.degrees[1] + 1):
                    derivatives[:, :, :, k, t] += (
                        basis_values_derivatives_v[t][:, :, :, s].unsqueeze(-1) * temp[s]
                    )

        normals = torch.linalg.cross(derivatives[:, :, :, 1, 0, :3], derivatives[:, :, :, 0, 1, :3])
        normals = torch.nn.functional.normalize(normals, dim=-1)

        normals = torch.cat(
            (normals, torch.zeros(tuple(normals.shape[:-1]) + (1,), device=device)), dim=-1
        )

        return derivatives[:, :, :, 0, 0], normals

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
