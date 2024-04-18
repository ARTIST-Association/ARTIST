import torch

class NURBSSurface(torch.nn.Module):

    def __init__(self, 
                 degree_x: int,
                 degree_y: int, 
                 output_dimension_x: int=32, 
                 output_dimension_y: int=32, 
                 dimension: int=3) -> torch.Tensor:
        super(NURBSSurface, self).__init__()
        self._dimension = dimension
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.evaluation_points_rows_x = torch.linspace(1e-5, 1.0-1e-5, steps=output_dimension_x, dtype=torch.float32)
        self.evaluation_points_columns_y = torch.linspace(1e-5, 1.0-1e-5, steps=output_dimension_y, dtype=torch.float32)
        self.evaluation_points =  torch.cartesian_prod(self.evaluation_points_rows_x, self.evaluation_points_columns_y)
        self.evaluation_points_x = self.evaluation_points[:, 0]
        self.evaluation_points_y = self.evaluation_points[:, 1]


    def forward(self, input):
        control_points, knot_vector_x, knot_vector_y = input
        control_points = control_points.unsqueeze(dim=0)

        # find span indices x direction (based on A2.1, page 68) 
        # add -1
        span_indices_x = torch.zeros(len(self.evaluation_points_x), dtype=torch.int64)
        for i, evaluation_point in enumerate(self.evaluation_points_x):
            if torch.isclose(evaluation_point, knot_vector_x[:, control_points.shape[1] + 1], atol=1e-5, rtol=1e-5):
                span_indices_x[i] = control_points.shape[1] - 1
                continue
            low = self.degree_x
            high = control_points.shape[1] + 1
            mid = (low + high) // 2
            while evaluation_point < knot_vector_x[:, mid] or evaluation_point >= knot_vector_x[:, mid + 1]:           
                if evaluation_point < knot_vector_x[:, mid]:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            span_indices_x[i] = mid

        evaluation_points_x = self.evaluation_points_x.squeeze(0)

        # find basis values x direction (based on A2.2, page 70)
        # basis_values_x = BasisFunction.apply(evaluation_points_x, knot_vector_x, span_indices_x.unsqueeze(-1), self.degree_x)

        # find span indices y direction (based on A2.1, page 68)
        # add - 1
        span_indices_y = torch.zeros(len(self.evaluation_points_y), dtype=torch.int64)
        for i, evaluation_point in enumerate(self.evaluation_points_y):
            if torch.isclose(evaluation_point, knot_vector_x[:, control_points.shape[2] + 1], atol=1e-5, rtol=1e-5):
                span_indices_y[i] = control_points.shape[2] - 1
                continue
            low = self.degree_y
            high = control_points.shape[2] + 1
            mid = (low + high) // 2
            while evaluation_point < knot_vector_y[:, mid] or evaluation_point >= knot_vector_y[:, mid + 1]:           
                if evaluation_point < knot_vector_y[:, mid]:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            span_indices_y[i] = mid
        
        evaluation_points_y = self.evaluation_points_y.squeeze(0)
        
        # find basis values y direction (based on A2.2, page 70)
        # basis_values_y = BasisFunction.apply(evaluation_points_y, knot_vector_y, span_indices_y.unsqueeze(-1), self.degree_y)

        # # added parentheses to make it work
        # pts = torch.stack([
        #     torch.stack([
        #         torch.stack([
        #             control_points[s,(span_indices_x.unsqueeze(0)[s,:]-(self.degree_x+l)),:,:][:,(span_indices_y.unsqueeze(0)[s,:]-(self.degree_y+r)),:]
        #             for r in range(self.degree_y+1)
        #         ])
        #         for l in range(self.degree_x+1)
        #     ]) 
        #     for s in range(knot_vector_x.size(0))
        # ])

        # surfaces = torch.sum((basis_values_x*pts)*basis_values_y, (1,2))
        # surfaces = surfaces[:,:,:,:self._dimension]

        control_point_weights = torch.zeros((control_points.shape[1],control_points.shape[2]) + (1,))
        control_points = torch.cat([control_points.squeeze(0), control_point_weights], dim=-1)

        nth_derivative = 1
        derivatives = torch.zeros(len(evaluation_points_x), nth_derivative + 1, nth_derivative + 1, control_points.shape[-1])

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
        basis_values_derivatives_x = BasisFunctionDerivatives.apply(evaluation_points_x, knot_vector_x.squeeze(0), span_indices_x, self.degree_x, dx)
        basis_values_derivatives_y = BasisFunctionDerivatives.apply(evaluation_points_y, knot_vector_y.squeeze(0), span_indices_y, self.degree_y, dy)
        
        temp = [torch.zeros((len(evaluation_points_x), control_points.shape[-1])) for _ in range(self.degree_y + 1)]
        for k in range(dx + 1): 
            for s in range(self.degree_y + 1):
                temp[s] = torch.zeros_like(temp[s])
                for r in range(self.degree_x + 1):
                    temp[s] += (basis_values_derivatives_x[k][r].squeeze(0).squeeze(0).unsqueeze(-1) * control_points[span_indices_x - self.degree_x + r, span_indices_y - self.degree_y + s])
            dd = min(nth_derivative - k, dy)
            for l in range(dd + 1):
                derivatives[:, k, l] = 0
                for s in range(self.degree_y + 1):
                    derivatives[:, k, l] += ((basis_values_derivatives_y[l][s].unsqueeze(-1) * temp[s])).squeeze(0).squeeze(0)

        normals = torch.linalg.cross(derivatives[:, 0, 1, :3], derivatives[:, 1, 0, :3])
        #normals /=torch.linalg.norm(normals, dim=1).unsqueeze(1)
        normals = torch.nn.functional.normalize(normals)
        
        return derivatives[:, 0, 0], normals


class BasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evaluation_points, knot_vector, span_indices, degree):
        ctx.save_for_backward(knot_vector)
        ctx.span_indices = span_indices
        ctx.evaluation_points = evaluation_points
        ctx.degree = degree

        evaluation_points = evaluation_points.squeeze(0)
        basis_values = [evaluation_points*0 for i in range(degree+1)]
        basis_values[0] = evaluation_points*0 + 1
        for k in range(1,degree+1):
            saved = (evaluation_points)*0.0
            for r in range(k):
                left = torch.stack([knot_vector[s, span_indices[s,:] + r + 1] for s in range(knot_vector.size(0))])
                right = torch.stack([knot_vector[s, span_indices[s,:] + 1 - k + r] for s in range(knot_vector.size(0))])
                temp = basis_values[r]/((left - evaluation_points) + (evaluation_points - right))
                temp = torch.where(((left - evaluation_points) + (evaluation_points - right))==0.0, evaluation_points*0+1e-8, temp)
                basis_values[r] = saved + (left - evaluation_points)*temp
                saved = (evaluation_points - right)*temp
            basis_values[k] = saved

        basis = torch.stack(basis_values).permute(1,0,2)
        ctx.basis_values = basis
        return basis

class BasisFunctionDerivatives(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evaluation_points, knot_vector, span, degree, nth_derivative=1):
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
        ders = [[torch.zeros(num_evaluation_points) for _ in range(degree + 1)] for _ in range(nth_derivative + 1)]
        for j in range(degree + 1):
            ders[0][j] = ndu[j][degree]
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
                ders[k][r] = d
                j = s1
                s1 = s2
                s2 = j

        r = degree
        for k in range(1, nth_derivative + 1):
            for j in range(degree + 1):
                ders[k][j] *= r
            r *= (degree - k)
        return ders



receiver_plane_x = 8.629666667
receiver_plane_y = 7.0

degree_x = 2
degree_y = 2
num_control_points_x = 7
num_control_points_y = 7

next_degree_x = degree_x + 1                                                          
next_degree_y = degree_y + 1 

origin = torch.tensor([0.0, 5.0, 0.0]) # heliostat position in field                                                                                    

control_points_shape = (num_control_points_x, num_control_points_y)                       
control_points = torch.zeros(
    control_points_shape + (3,),                                                  
)

origin_offsets_x = torch.linspace(
    -receiver_plane_x / 2, receiver_plane_x / 2, num_control_points_x)
origin_offsets_y = torch.linspace(
    -receiver_plane_y / 2, receiver_plane_y / 2, num_control_points_y)
origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
origin_offsets = torch.hstack((
    origin_offsets,
    torch.zeros((len(origin_offsets), 1)),
))
control_points = (origin + origin_offsets).reshape(control_points.shape)                      

knots_x = torch.zeros(num_control_points_x + next_degree_x)                                                                                              
num_knot_vals = len(knots_x[degree_x:-degree_x])
knot_vals = torch.linspace(0, 1, num_knot_vals)
knots_x[:degree_x] = 0
knots_x[degree_x:-degree_x] = knot_vals
knots_x[-degree_x:] = 1

knots_y = torch.zeros(num_control_points_y + next_degree_y)                                                                                        
num_knot_vals = len(knots_y[degree_y:-degree_y])
knot_vals = torch.linspace(0, 1, num_knot_vals)
knots_y[:degree_y] = 0
knots_y[degree_y:-degree_y] = knot_vals
knots_y[-degree_y:] = 1

nurbs = NURBSSurface(degree_x, degree_y, 2, 2)
input = (control_points, knots_x.unsqueeze(0), knots_y.unsqueeze(0))
surface_points, surface_normals = nurbs(input)

print('points', surface_points)
print(surface_normals)
