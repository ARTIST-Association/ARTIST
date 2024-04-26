from typing import Tuple
import torch

class NURBSSurface(torch.nn.Module):

    def __init__(self, 
                 degree_x: int,
                 degree_y: int, 
                 evaluation_points_x: torch.Tensor,
                 evaluation_points_y: torch.Tensor,
                 control_points: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Initialize a nurbs surface.

        Parameters
        ----------
        degree_x : int
            The spline degree in x direction.
        degree_y : int
            The spline degree in y direction.
        evaluation_points_x : torch.Tensor
            The evaluation points in x direction.
        evaluation_points_y : torch.Tensor
            The evaluation points in y direction.
        control_points : torch.Tensor
            The control_points.
        """
        super(NURBSSurface, self).__init__()
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.evaluation_points_x = evaluation_points_x
        self.evaluation_points_y = evaluation_points_y
        self.control_points = control_points
        self.knot_vector_x, self.knot_vector_y = self.calculate_knots()

    def calculate_knots(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the knot vectors in x and y direction.

        Returns
        -------
        torch.Tensor
            The knots in x direction.
        torch.Tensor
            The knots in y direction.
        """
        num_control_points_x = self.control_points.shape[0]
        num_control_points_y = self.control_points.shape[1]

        knots_x = torch.zeros(num_control_points_x + self.degree_x + 1)                                                                                              
        num_knot_vals = len(knots_x[self.degree_x:-self.degree_x])
        knot_vals = torch.linspace(0, 1, num_knot_vals)
        knots_x[:self.degree_x] = 0
        knots_x[self.degree_x:-self.degree_x] = knot_vals
        knots_x[-self.degree_x:] = 1

        knots_y = torch.zeros(num_control_points_y + self.degree_x + 1)                                                                                        
        num_knot_vals = len(knots_y[self.degree_y:-self.degree_y])
        knot_vals = torch.linspace(0, 1, num_knot_vals)
        knots_y[:self.degree_y] = 0
        knots_y[self.degree_y:-self.degree_y] = knot_vals
        knots_y[-self.degree_y:] = 1

        return knots_x, knots_y

    def find_span(self, 
                  degree: int,
                  evaluation_points: torch.Tensor,
                  knot_vector: torch.Tensor,
                  control_points: torch.Tensor,
                  ) -> torch.Tensor:
        """
        Determine the knot span index for given evaluation points.

        Later on the basis functions are evaluated, some of them are
        identical to zero and therefore it would be a waste to compute 
        them. That is why first the knot span in which the evaluation 
        point lies is computed using this function.
        See `The NURBS Book` page 68 for reference.

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
            while evaluation_point < knot_vector[mid] or evaluation_point >= knot_vector[mid + 1]:           
                if evaluation_point < knot_vector[mid]:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            span_indices[i] = mid
        
        return span_indices
        
    def basis_function_and_derivatives(self,
                                       evaluation_points: torch.Tensor, 
                                       knot_vector: torch.Tensor,
                                       span: torch.Tensor,
                                       degree: int,
                                       nth_derivative: int=1) -> torch.Tensor:
        """
        Compute the nonzero derivatives of the basis functions up to the nth-derivative.

        See `The NURBS Book` page 72 for reference.

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
            Specifies how many derivatives are calculated.
        
        Returns
        -------
        torch.Tensor
            The derivatives of the basis function.
        """
        num_evaluation_points = len(evaluation_points)
        ndu = [[torch.zeros(num_evaluation_points) for _ in range(degree + 1)] for _ in range(degree + 1)]
        ndu[0][0] = torch.ones_like(ndu[0][0])
        left = [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
        right = [torch.zeros(num_evaluation_points) for _ in range(degree + 1)]
        for j in range(1, degree + 1):
            left[j] = evaluation_points - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - evaluation_points
            saved = torch.zeros(num_evaluation_points)
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        derivatives = [[torch.zeros(num_evaluation_points) for _ in range(degree + 1)] for _ in range(nth_derivative + 1)]
        for j in range(degree + 1):
            derivatives[0][j] = ndu[j][degree]
        a = [[torch.zeros(num_evaluation_points) for _ in range(degree + 1)] for _ in range(2)]
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
                    a[s2][j] = ((a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j])
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
            r *= (degree - k)
        return derivatives

    def calculate_surface_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # find span indices x direction (based on A2.1, page 68) 
        span_indices_x = self.find_span(self.degree_x, self.evaluation_points_x, self.knot_vector_x, self.control_points)

        # find span indices y direction (based on A2.1, page 68)
        span_indices_y = self.find_span(self.degree_y, self.evaluation_points_y, self.knot_vector_y, self.control_points)
        
        control_point_weights = torch.ones((self.control_points.shape[0], self.control_points.shape[1]) + (1,))
        control_points = torch.cat([self.control_points, control_point_weights], dim=-1)

        derivatives = torch.zeros(len(self.evaluation_points_x), nth_derivative + 1, nth_derivative + 1, control_points.shape[-1])

        # find surface points and normals (based on A3.6, page 111)
        dx = min(nth_derivative, self.degree_x)
        for k in range(self.degree_x + 1, nth_derivative + 1):
            for l in range(nth_derivative - k + 1):
                derivatives[:, k, l] = 0
        dy = min(nth_derivative, self.degree_y)
        for l in range(self.degree_y + 1, nth_derivative + 1):
            for k in range(nth_derivative - l + 1):
                derivatives[:, k, l] = 0
        
        # find derivatives of basis functions (based on A2.3, page 72)
        basis_values_derivatives_x = self.basis_function_and_derivatives(self.evaluation_points_x, self.knot_vector_x, span_indices_x, self.degree_x, dx)
        basis_values_derivatives_y = self.basis_function_and_derivatives(self.evaluation_points_y, self.knot_vector_y, span_indices_y, self.degree_y, dy)
        
        temp = [torch.zeros((len(self.evaluation_points_x), control_points.shape[-1])) for _ in range(self.degree_y + 1)]
        for k in range(dx + 1): 
            for s in range(self.degree_y + 1):
                temp[s] = torch.zeros_like(temp[s])
                for r in range(self.degree_x + 1):
                    temp[s] += (basis_values_derivatives_x[k][r].unsqueeze(-1) * control_points[span_indices_x - self.degree_x + r, span_indices_y - self.degree_y + s])
            dd = min(nth_derivative - k, dy)
            for l in range(dd + 1):
                derivatives[:, k, l] = 0
                for s in range(self.degree_y + 1):
                    derivatives[:, k, l] += ((basis_values_derivatives_y[l][s].unsqueeze(-1) * temp[s]))

        normals = torch.linalg.cross(derivatives[:, 1, 0, :3], derivatives[:, 0, 1, :3])
        normals = torch.nn.functional.normalize(normals)

        normals = torch.cat((normals, torch.zeros(normals.shape[0], 1)), dim = 1)
        
        return derivatives[:, 0, 0], normals
