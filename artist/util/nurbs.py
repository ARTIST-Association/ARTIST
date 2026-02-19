import torch

from artist.util import index_mapping, utils
from artist.util.environment_setup import get_device


class NURBSSurfaces(torch.nn.Module):
    """
    Implement differentiable NURBS for surface representations.

    This implementation can be used to create multiple separate NURBS surfaces at the same time
    and they are handled in parallel.

    Attributes
    ----------
    degrees : torch.Tensor
        The spline degrees in u and then in v direction.
        Tensor of shape [2].
    control_points : torch.Tensor
        The control points.
        Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
    uniform : bool
        Indicates wether the NURBS are uniform or not.
    number_of_surfaces : int
        The number of NURBS surfaces processed in parallel (number of heliostats).
    number_of_facets_per_surface : int
        The number of facets per single NURBS surface.
    knot_vectors_u : torch.Tensor
        The knot vectors in the u direction.
        Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction + degree_u_direction + 1].
    knot_vectors_v : torch.Tensor
        The knot vectors in the v direction.
        Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_v_direction + degree_v_direction + 1].

    Methods
    -------
    calculate_uniform_knot_vectors()
        Calculate the knot vectors for all surfaces in one direction.
    find_spans()
        Determine the knot spans in one direction.
    basis_functions_and_derivatives()
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.
    calculate_surface_points_and_normals()
        Calculate the surface points and normals of the NURBS surfaces.
    forward()
        Specify the forward operation of the NURBS, i.e., calculate the surface points and normals.
    """

    def __init__(
        self,
        degrees: torch.Tensor,
        control_points: torch.Tensor,
        uniform: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize NURBS surfaces.

        NURBS stands for Non-Uniform Rational B-Spline. NURBS allow for an efficient and precise reconstruction
        of the imperfect heliostat surfaces in the digital twin. This implementation of the NURBS is
        differentiable. The NURBS surfaces require a degree in two directions and control points. These parameters
        are used to create the NURBS surface. For more details, see the NURBS tutorial.

        Parameters
        ----------
        degrees : torch.Tensor
            The spline degrees in u and then in v direction.
            Tensor of shape [2].
        control_points : torch.Tensor
            The control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        uniform : bool
            Indicates wether the NURBS are uniform or not (default is True (uniform)).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.degrees = degrees
        self.control_points = control_points
        self.uniform = uniform
        self.number_of_surfaces = self.control_points.shape[
            index_mapping.nurbs_surfaces
        ]
        self.number_of_facets_per_surface = self.control_points.shape[
            index_mapping.nurbs_facets
        ]
        self.knot_vectors_u = self.calculate_uniform_knot_vectors(
            direction=index_mapping.nurbs_u, device=device
        )
        self.knot_vectors_v = self.calculate_uniform_knot_vectors(
            direction=index_mapping.nurbs_v, device=device
        )

    def calculate_uniform_knot_vectors(
        self,
        direction: int,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """
        Calculate the knot vectors for all surfaces in one direction.

        For our application, only uniform knot vectors are required.
        The knots range from zero to one and are distributed uniformly.
        The first knot (0) and the last knot (1) have full multiplicity,
        i.e., they are repeated as often as specified by the degree.
        This means the NURBS start and end in a control point.

        Parameters
        ----------
        direction : int
            The NURBS surface direction u or v.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The knot vectors.
        """
        device = get_device(device=device)

        degree = self.degrees[direction].item()

        knot_vector = torch.zeros(
            (
                self.control_points.shape[
                    index_mapping.nurbs_control_points_start + direction
                ]
                + degree
                + 1
            ),
            device=device,
        )
        number_of_knot_values = knot_vector[degree:-degree].shape[
            index_mapping.nurbs_knots_unbatched
        ]
        knot_vector[:degree] = 0
        knot_vector[degree:-degree] = torch.linspace(
            0, 1, number_of_knot_values, device=device
        )
        knot_vector[-degree:] = 1

        knot_vectors = knot_vector.unsqueeze(
            index_mapping.nurbs_knots_unbatched
        ).repeat(
            self.number_of_surfaces,
            self.number_of_facets_per_surface,
            index_mapping.nurbs_facets,
        )

        return knot_vectors

    def find_spans(
        self,
        direction: int,
        evaluation_points: torch.Tensor,
        knot_vectors: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Determine the knot spans in one direction.

        To generate NURBS, the basis functions must be evaluated. However, some basis functions may be zero. To improve
        computational efficiency, basis functions that are zero are not computed. Therefore, the knot spans in which the
        evaluation point lies is first computed using this function.
        See `The NURBS Book` p. 68 for reference.
        If the knot vector is uniform, the spans can be computed more efficiently.

        Parameters
        ----------
        direction : int
            The NURBS surface direction u or v.
        evaluation_points : torch.Tensor
            The evaluation points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 2].
        knot_vectors : torch.Tensor
            The knot vector for the NURBS surfaces in a single direction.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_one_direction + degree_one_direction + 1].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The knot spans.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points].
        """
        device = get_device(device=device)
        degree = self.degrees[direction].item()
        evaluation_points = evaluation_points[:, :, :, direction]

        if self.uniform:
            unique_knots = torch.unique(
                knot_vectors, dim=index_mapping.nurbs_knots_batched
            )

            spans = (
                torch.floor(
                    evaluation_points
                    * (unique_knots.shape[index_mapping.nurbs_knots_batched] - 1)
                ).long()
                + degree
            )

        else:
            number_of_knots = (
                knot_vectors.shape[index_mapping.nurbs_knots_batched] - degree - 1
            )

            valid_spans = []
            for knot_index in range(degree, number_of_knots):
                left = knot_vectors[:, :, knot_index]
                right = knot_vectors[:, :, knot_index + 1]
                valid_spans.append((left, right))

            lefts = torch.stack(
                [span[index_mapping.nurbs_span_lower] for span in valid_spans],
                dim=index_mapping.nurbs_knots_batched,
            )
            rights = torch.stack(
                [span[index_mapping.nurbs_span_upper] for span in valid_spans],
                dim=index_mapping.nurbs_knots_batched,
            )

            in_span = (
                evaluation_points.unsqueeze(-1)
                >= lefts.unsqueeze(index_mapping.nurbs_spans)
            ) & (
                evaluation_points.unsqueeze(-1)
                < rights.unsqueeze(index_mapping.nurbs_spans)
            )
            is_last_knot = torch.isclose(
                evaluation_points,
                knot_vectors[:, :, number_of_knots],
                atol=1e-5,
                rtol=1e-5,
            )
            spans = in_span.int().argmax(dim=-1) + degree
            spans = torch.where(
                is_last_knot, torch.full_like(spans, number_of_knots - 1), spans
            )

        return spans

    def basis_functions_and_derivatives(
        self,
        direction: int,
        evaluation_points: torch.Tensor,
        knot_vectors: torch.Tensor,
        spans: torch.Tensor,
        nth_derivative: int = 1,
        device: torch.device | None = None,
    ) -> list[list[torch.Tensor]]:
        """
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.

        See `The NURBS Book` p. 72 for reference.

        Parameters
        ----------
        direction : int
            The NURBS surface direction u or v.
        evaluation_points : torch.Tensor
            The evaluation points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 2].
        knot_vectors : torch.Tensor
            Contains all the knots of the NURBS surfaces.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_one_direction + degree_one_direction + 1].
        spans : torch.Tensor
            The knot spans.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points].
        nth_derivative : int
            Specifies how many derivatives are calculated (default is 1).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        list[list[torch.Tensor]]
            The derivatives of the basis functions.
        """
        device = get_device(device=device)

        degree = self.degrees[direction].item()

        evaluation_points = evaluation_points[:, :, :, direction]
        num_evaluation_points = evaluation_points.shape[
            index_mapping.nurbs_evaluation_points
        ]

        # Introduce `ndu` to store the basis functions (called "n" in The NURBS book) and the knot differences (du).
        ndu = torch.zeros(
            (
                degree + 1,
                degree + 1,
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                num_evaluation_points,
            ),
            device=device,
        )
        ndu[
            index_mapping.nurbs_ndu_basis_index_0, index_mapping.nurbs_ndu_basis_index_0
        ] = 1.0

        left = torch.zeros(
            (
                degree + 1,
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                num_evaluation_points,
            ),
            device=device,
        )
        right = torch.zeros(
            (
                degree + 1,
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                num_evaluation_points,
            ),
            device=device,
        )

        for j in range(1, degree + 1):
            left[j] = evaluation_points - torch.gather(knot_vectors, 2, spans - j + 1)
            right[j] = torch.gather(knot_vectors, 2, spans + j) - evaluation_points
            saved = torch.zeros(
                (
                    self.number_of_surfaces,
                    self.number_of_facets_per_surface,
                    num_evaluation_points,
                ),
                device=device,
            )
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                # Introduce `tmp` to temporarily store result.
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        derivatives = [
            [
                torch.zeros(
                    (
                        self.number_of_surfaces,
                        self.number_of_facets_per_surface,
                        num_evaluation_points,
                    ),
                    device=device,
                )
                for _ in range(degree + 1)
            ]
            for _ in range(nth_derivative + 1)
        ]
        for j in range(degree + 1):
            derivatives[index_mapping.basis_function_derivative_order][j] = ndu[j][
                degree
            ]
        # `a` stores (in alternating fashion) the two most recently computed rows a_k,j and a_k-1,j.
        a = [
            [
                torch.zeros(
                    (
                        self.number_of_surfaces,
                        self.number_of_facets_per_surface,
                        num_evaluation_points,
                    ),
                    device=device,
                )
                for _ in range(degree + 1)
            ]
            for _ in range(2)
        ]
        for r in range(degree + 1):
            s1 = 0
            s2 = 1
            a[0][0] = torch.ones_like(a[0][0], device=device)
            for k in range(1, nth_derivative + 1):
                d = torch.zeros(
                    (
                        self.number_of_surfaces,
                        self.number_of_facets_per_surface,
                        num_evaluation_points,
                    ),
                    device=device,
                )
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

    def _batched_gather_control_points(
        self,
        control_points: torch.Tensor,
        index_u: torch.Tensor,
        index_v: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Gathers control points using batched 2D indices.

        Parameters
        ----------
        control_points : torch.Tensor
            The control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 4].
        index_u : torch.Tensor
            The indices in u direction.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points].
        index_v : torch.Tensor
            The indices in v direction.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The gathered control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        """
        device = get_device(device=device)

        batch_index = (
            torch.arange(self.number_of_surfaces, device=device)
            .view(self.number_of_surfaces, 1, 1)
            .expand(
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                index_u.shape[index_mapping.nurbs_evaluation_points],
            )
        )
        facet_index = (
            torch.arange(self.number_of_facets_per_surface, device=device)
            .view(1, self.number_of_facets_per_surface, 1)
            .expand(
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                index_u.shape[index_mapping.nurbs_evaluation_points],
            )
        )

        gathered = control_points[batch_index, facet_index, index_u, index_v]
        return gathered

    def calculate_surface_points_and_normals(
        self,
        evaluation_points: torch.Tensor,
        canting: torch.Tensor,
        facet_translations: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the surface points and normals of the NURBS surfaces.

        Parameters
        ----------
        evaluation_points : torch.Tensor
            The evaluation points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        torch.Tensor
            The surface normals.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        """
        device = get_device(device=device)

        nth_derivative = 1

        # Find the spans in u direction (based on A2.1, p. 68).
        spans_u = self.find_spans(
            direction=index_mapping.nurbs_u,
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_u,
            device=device,
        )

        # Find the spans in v direction (based on A2.1, p. 68).
        spans_v = self.find_spans(
            direction=index_mapping.nurbs_v,
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_v,
            device=device,
        )

        control_point_weights = torch.ones(
            (
                self.number_of_surfaces,
                self.number_of_facets_per_surface,
                self.control_points.shape[index_mapping.nurbs_control_points_u],
                self.control_points.shape[index_mapping.nurbs_control_points_v],
                1,
            ),
            device=device,
        )
        control_points = torch.cat(
            [self.control_points, control_point_weights],
            dim=index_mapping.nurbs_control_points,
        )

        derivatives = torch.zeros(
            self.number_of_surfaces,
            self.number_of_facets_per_surface,
            evaluation_points.shape[index_mapping.nurbs_evaluation_points],
            nth_derivative + 1,
            nth_derivative + 1,
            control_points.shape[index_mapping.nurbs_control_points],
            device=device,
        )

        # Find minimum of `nth_derivative` and degree, will be used to specify how many partial derivatives will be
        # computed.
        du = min(nth_derivative, self.degrees[index_mapping.nurbs_u])
        for k in range(self.degrees[index_mapping.nurbs_u] + 1, nth_derivative + 1):
            for t in range(nth_derivative - k + 1):
                derivatives[:, :, :, k, t] = 0
        dv = min(nth_derivative, self.degrees[index_mapping.nurbs_v])
        for t in range(self.degrees[index_mapping.nurbs_v] + 1, nth_derivative + 1):
            for k in range(nth_derivative - t + 1):
                derivatives[:, :, :, k, t] = 0

        # Find derivatives of basis functions (based on A2.3, p. 72).
        basis_values_derivatives_u = self.basis_functions_and_derivatives(
            direction=index_mapping.nurbs_u,
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_u,
            spans=spans_u,
            nth_derivative=du,
            device=device,
        )
        basis_values_derivatives_v = self.basis_functions_and_derivatives(
            direction=index_mapping.nurbs_v,
            evaluation_points=evaluation_points,
            knot_vectors=self.knot_vectors_v,
            spans=spans_v,
            nth_derivative=dv,
            device=device,
        )

        # Find surface points and normals (based on A3.6, p. 111).
        # `temp` stores the vector/matrix product of the basis value derivatives and the control points.
        temp = [
            torch.zeros(
                (
                    self.number_of_surfaces,
                    self.number_of_facets_per_surface,
                    evaluation_points.shape[index_mapping.nurbs_evaluation_points],
                    control_points.shape[index_mapping.nurbs_control_points],
                ),
                device=device,
            )
            for _ in range(self.degrees[index_mapping.nurbs_v] + 1)
        ]
        for k in range(du + 1):
            for s in range(self.degrees[index_mapping.nurbs_v] + 1):
                temp[s] = torch.zeros_like(temp[s], device=device)
                for r in range(self.degrees[index_mapping.nurbs_u] + 1):
                    bu = basis_values_derivatives_u[k][r].unsqueeze(-1)
                    index_u = spans_u - self.degrees[index_mapping.nurbs_u] + r
                    index_v = spans_v - self.degrees[index_mapping.nurbs_v] + s
                    gathered_control_points = self._batched_gather_control_points(
                        control_points=control_points,
                        index_u=index_u,
                        index_v=index_v,
                        device=device,
                    )
                    temp[s] += bu * gathered_control_points

            dd = min(nth_derivative - k, dv)
            for t in range(dd + 1):
                derivatives[:, :, :, k, t] = 0
                for s in range(self.degrees[index_mapping.nurbs_v] + 1):
                    derivatives[:, :, :, k, t] += (
                        basis_values_derivatives_v[t][s].unsqueeze(-1) * temp[s]
                    )

        surface_normals = torch.linalg.cross(
            derivatives[
                :,
                :,
                :,
                index_mapping.nurbs_derivative_order_1,
                index_mapping.nurbs_derivative_order_0,
                :3,
            ],
            derivatives[
                :,
                :,
                :,
                index_mapping.nurbs_derivative_order_0,
                index_mapping.nurbs_derivative_order_1,
                :3,
            ],
        )

        surface_points_homogenous = derivatives[
            :,
            :,
            :,
            index_mapping.nurbs_derivative_order_0,
            index_mapping.nurbs_derivative_order_0,
        ]

        homogenous_weights = surface_points_homogenous[:, :, :, 3:4]
        surface_points = (
            surface_points_homogenous[:, :, :, : index_mapping.slice_fourth_dimension]
            / homogenous_weights
        )

        surface_points = torch.cat(
            (
                surface_points,
                torch.ones(
                    tuple(surface_points.shape[:3]) + (1,),
                    device=device,
                ),
            ),
            dim=index_mapping.nurbs_normals,
        )

        surface_normals = torch.nn.functional.normalize(
            surface_normals, dim=index_mapping.nurbs_normals
        )

        surface_normals = torch.cat(
            (
                surface_normals,
                torch.zeros(
                    tuple(surface_normals.shape[: index_mapping.nurbs_normals]) + (1,),
                    device=device,
                ),
            ),
            dim=index_mapping.nurbs_normals,
        )

        if canting is not None:
            canted_surface_points = utils.perform_canting(
                canting_angles=canting, data=surface_points, device=device
            )
            transformed_surface_points = (
                canted_surface_points
                + facet_translations.reshape(
                    self.number_of_surfaces, self.number_of_facets_per_surface, 1, 4
                )
            )
            transformed_surface_normals = utils.perform_canting(
                canting_angles=canting, data=surface_normals, device=device
            )
            return transformed_surface_points, transformed_surface_normals

        return surface_points, surface_normals

    def forward(
        self,
        evaluation_points: torch.Tensor,
        canting: torch.Tensor,
        facet_translations: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Specify the forward operation of the NURBS, i.e., calculate the surface points and normals.

        Parameters
        ----------
        evaluation_points : torch.Tensor
            The evaluation points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The surface points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        torch.Tensor
            The surface normals.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        """
        device = get_device(device=device)

        return self.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points,
            canting=canting,
            facet_translations=facet_translations,
            device=device,
        )
