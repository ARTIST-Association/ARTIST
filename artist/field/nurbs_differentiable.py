import torch
from torch.autograd import Variable

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
        self.evaluation_points_x = torch.linspace(1e-5, 1.0-1e-5, steps=output_dimension_x, dtype=torch.float32)
        self.evaluation_points_y = torch.linspace(1e-5, 1.0-1e-5, steps=output_dimension_y, dtype=torch.float32)

    def forward(self, input):
        control_points, knot_vector_x, knot_vector_y = input
        control_points = control_points.unsqueeze(dim=0)

        ######## Test #########
        # self.degree_x = 2
        # knot_vector_x = torch.tensor([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]).unsqueeze(dim=0)
        # self.evaluation_points_x = torch.tensor([0.1746])
        #######################


        # normalize knot vector
        knot_vector_x_ = torch.cumsum(torch.where(knot_vector_x<0.0, knot_vector_x*0+1e-4, knot_vector_x), dim=1)
        knot_vector_x = (knot_vector_x_ - knot_vector_x_[:, 0].unsqueeze(-1)) / (knot_vector_x_[:, -1].unsqueeze(-1) - knot_vector_x_[:, 0].unsqueeze(-1))
        knot_vector_y_ = torch.cumsum(torch.where(knot_vector_y<0.0, knot_vector_y*0+1e-4, knot_vector_y), dim=1)
        knot_vector_y = (knot_vector_y_ - knot_vector_y_[:, 0].unsqueeze(-1)) / (knot_vector_y_[:, -1].unsqueeze(-1) - knot_vector_y_[:, 0].unsqueeze(-1))

        # find span indices x direction (based on A2.1) 
        span_indices_x = torch.empty(
            len(self.evaluation_points_x),
            dtype=torch.int64
        )
        not_upper_span_indices = \
            self.evaluation_points_x != knot_vector_x.squeeze(0)[control_points.shape[1]]
        span_indices_x[~not_upper_span_indices] = control_points.shape[1] - 1
        spans = torch.searchsorted(
            knot_vector_x.squeeze(0),
            self.evaluation_points_x[not_upper_span_indices],
            right=True,
        ) - 1
        span_indices_x[not_upper_span_indices] = spans

        # eps = 1e-5
        # if abs(self.evaluation_points_x - knot_vector_x[:, control_points.shape[1] + 1]) < eps:
        #     return control_points.shape[1] - 1
        # low = self.degree_x
        # high = control_points.shape[1] + 1
        # mid = (low + high) // 2
        # while self.evaluation_points_x < knot_vector_x[mid] - eps or self.evaluation_points_x >= knot_vector_x[mid + 1] + eps:
        #     if self.evaluation_points_x < knot_vector_x[mid] - eps:
        #         high = mid
        #     else:
        #         low = mid
        #     mid = (low + high) // 2
        # span_indices_x = mid




        #evaluation_points_x = self.evaluation_points_x.unsqueeze(0)
        #span_indices_x = torch.stack([torch.min(torch.where((evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1))>1e-8, evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1), (evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.degree_x for s in range(knot_vector_x.size(0))])
        evaluation_points_x = self.evaluation_points_x.squeeze(0)

        # find basis values x direction (based on A2.2)
        basis_values_x = BasisFunction.apply(evaluation_points_x, knot_vector_x, span_indices_x, self.degree_x).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)

        # find span indices y direction (based on A2.1) 
        
        span_indices_y = torch.empty(
            len(self.evaluation_points_y),
            dtype=torch.int64
        )
        not_upper_span_indices = \
            self.evaluation_points_y != knot_vector_y.squeeze(0)[control_points.shape[2]]
        span_indices_y[~not_upper_span_indices] = control_points.shape[2] - 1
        spans = torch.searchsorted(
            knot_vector_y.squeeze(0),
            self.evaluation_points_y[not_upper_span_indices],
            right=True,
        ) - 1
        span_indices_y[not_upper_span_indices] = spans
        
        #evaluation_points_y = self.evaluation_points_y.unsqueeze(0)
        #span_indices_y = torch.stack([torch.min(torch.where((evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1))>1e-8, evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1), (evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.degree_y for s in range(knot_vector_y.size(0))])
        evaluation_points_y = self.evaluation_points_y.squeeze(0)
        
        # find basis values y direction (based on A2.2)
        basis_values_y = BasisFunction.apply(evaluation_points_y, knot_vector_y, span_indices_y, self.degree_y).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)

        # added parentheses to make it work
        pts = torch.stack([
            torch.stack([
                torch.stack([
                    control_points[s,(span_indices_x.unsqueeze(0)[s,:]-(self.degree_x+l)),:,:][:,(span_indices_y.unsqueeze(0)[s,:]-(self.degree_y+r)),:]
                    for r in range(self.degree_y+1)
                ])
                for l in range(self.degree_x+1)
            ]) 
            for s in range(knot_vector_x.size(0))
        ])

        surfaces = torch.sum((basis_values_x*pts)*basis_values_y, (1,2))
        surfaces = surfaces[:,:,:,:self._dimension]

        
        # find derivatives of basis values in x and y direction (based on A2.3)
        nth_derivative = 1
        derivatives = torch.empty(len(evaluation_points_x), nth_derivative + 1, nth_derivative + 1, control_points.shape[-1])
        basis_values_derivatives_x = BasisFunctionDerivatives.apply(evaluation_points_x, knot_vector_x, span_indices_x, self.degree_x, nth_derivative)
        basis_values_derivatives_y = BasisFunctionDerivatives.apply(evaluation_points_y, knot_vector_y, span_indices_y, self.degree_y, nth_derivative)
        
        # find surface normals
        dx = min(nth_derivative, self.degree_x)
        for k in range(self.degree_x + 1, nth_derivative + 1):
            for l in range(nth_derivative - k + 1):
                derivatives[:, k, l] = 0
        dy = min(nth_derivative, self.degree_y)
        for l in range(self.degree_y + 1, nth_derivative + 1):
            for k in range(nth_derivative - l + 1):
                derivatives[:, k, l] = 0
        temp = [torch.empty((len(evaluation_points_x), control_points.shape[-1])) for _ in range(self.degree_y + 1)]
        for k in range(dx + 1): 
            for s in range(self.degree_y + 1):
                temp[s] = torch.zeros_like(temp[s]).unsqueeze(0)
                for r in range(self.degree_x + 1):
                    temp[s] += basis_values_derivatives_x[k][r] * control_points.squeeze(0)[span_indices_x - self.degree_x - r, span_indices_y - self.degree_y - s]
            dd = min(nth_derivative - k, dy)
            for l in range(dd + 1):
                derivatives[:, k, l] = 0
                for s in range(self.degree_y + 1):
                    derivatives[:, k, l] += (basis_values_derivatives_y[l][s] * temp[s])
        return surfaces, derivatives


class BasisFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, evaluation_points, knot_vector, span_indices, degree):
        ctx.save_for_backward(knot_vector)
        ctx.span_indices = span_indices
        ctx.evaluation_points = evaluation_points
        ctx.degree = degree

        evaluation_points = evaluation_points.squeeze(0)
        basis_values = torch.empty(len(evaluation_points), degree + 1)
        basis_values[:, 0] = 1
        left = torch.empty(basis_values.shape)
        right = torch.empty(basis_values.shape)
        for j in range(1, degree + 1):
            left[:, j] = evaluation_points - knot_vector[:, span_indices + 1 - j]
            # add -1
            right[:, j] = knot_vector[:, span_indices + j - 1] - evaluation_points
            saved = torch.zeros(len(evaluation_points))
            for r in range(0, j):
                temp = basis_values[:, r] / (right[:, r + 1] + left[:, j - r])
                basis_values[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            basis_values[:, j] = saved
        return basis_values

        # evaluation_points = evaluation_points.squeeze(0)
        # basis_values = [evaluation_points*0 for i in range(degree+1)]
        # basis_values[0] = evaluation_points*0 + 1
        # for k in range(1,degree+1):
        #     saved = (evaluation_points)*0.0
        #     for r in range(k):
        #         left = torch.stack([knot_vector[s, span_indices[s,:] + r + 1] for s in range(knot_vector.size(0))])
        #         right = torch.stack([knot_vector[s, span_indices[s,:] + 1 - k + r] for s in range(knot_vector.size(0))])
        #         temp = basis_values[r]/((left - evaluation_points) + (evaluation_points - right))
        #         temp = torch.where(((left - evaluation_points) + (evaluation_points - right))==0.0, evaluation_points*0+1e-8, temp)
        #         basis_values[r] = saved + (left - evaluation_points)*temp
        #         saved = (evaluation_points - right)*temp
        #     basis_values[k] = saved

        # basis = torch.stack(basis_values).permute(1,0,2)
        # ctx.basis_values = basis
        # return basis

    @staticmethod
    def backward(ctx, grad_output):
        knot_vector = ctx.saved_tensors[0]
        span_indices = ctx.span_indices
        degree = ctx.degree
        basis_values = ctx.basis_values
        evaluation_points = ctx.evaluation_points

        UList = torch.stack([knot_vector[s, span_indices[s,:]] for s in range(knot_vector.size(0))])

        derivatives_basis_values = [grad_output[:,0,:]*0 for i in range(degree+1)]
        derivatives_basis_values[0] = grad_output[:,0,:]*0 + 0.5*UList*basis_values[:,0,:]

        for k in reversed(range(1,degree+1)):
            temp = derivatives_basis_values[k]*grad_output[:,k,:]
            for r in reversed(range(k)):
                left = torch.stack([knot_vector[s, span_indices[s,:] + r + 1] for s in range(knot_vector.size(0))])
                right = torch.stack([knot_vector[s, span_indices[s,:] + 1 - k + r] for s in range(knot_vector.size(0))])
                temp = temp*(evaluation_points-right)
                derivatives_basis_values[r] += (left - evaluation_points)*temp
                derivatives_basis_values[r] = derivatives_basis_values[r]/((left - evaluation_points) + (evaluation_points - right))

        derivatives = torch.stack(derivatives_basis_values).permute(1,0,2)

        dU = knot_vector*0
        for s in range(knot_vector.size(0)):
            for k in range(0, degree + 1):
                dU[s, :].scatter_(-1, (span_indices[s,:] + k).type_as(span_indices), derivatives[s, k, :], reduce='add')
        dU = dU*knot_vector

        return Variable(knot_vector*0), Variable(dU), Variable(knot_vector*0), None
    

class BasisFunctionDerivatives(torch.autograd.Function):

    @staticmethod
    def forward(ctx, evaluation_points, knot_vector, span, degree, nth_derivative=1):
        num_evaluation_points = len(evaluation_points)
        ndu = [[torch.empty(num_evaluation_points) for _ in range(degree + 1)] for _ in range(degree + 1)]
        ndu[0][0] = torch.ones_like(ndu[0][0])
        left = [torch.empty(num_evaluation_points) for _ in range(degree + 1)]
        right = [torch.empty(num_evaluation_points) for _ in range(degree + 1)]
        for j in range(1, degree + 1):
            left[j] = evaluation_points - knot_vector[:, span + 1 - j]
            #add - 1
            right[j] = knot_vector[:, span + j -1] - evaluation_points
            saved = torch.zeros(num_evaluation_points)
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                tmp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            ndu[j][j] = saved
        ders = [[torch.empty(num_evaluation_points) for _ in range(degree + 1)] for _ in range(nth_derivative + 1)]
        for j in range(degree + 1):
            ders[0][j] = ndu[j][degree]
        a = [[torch.empty(num_evaluation_points) for _ in range(degree + 1)] for _ in range(2)]
        for r in range(degree + 1):
            s1 = 0
            s2 = 1
            a[0][0] = torch.ones_like(a[0][0])
            for k in range(1, nth_derivative + 1):
                d = torch.zeros(num_evaluation_points).unsqueeze(0).unsqueeze(0)
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
            r *= degree - k
        return ders
            

control_points_shape = (3, 3)                        
control_points = torch.empty(control_points_shape + (3,))   
knots_x = torch.tensor([0, 0, 0.1, 0.2, 0.3, 1]).unsqueeze(dim=0)
knots_y = torch.tensor([0, 0, 0.1, 0.2, 0.3, 1]).unsqueeze(dim=0)

knots_x = torch.zeros(3 + 2 + 1)                                 # num_control_points + order == num_control_points + degree + 1                                                             # fill knot vector with full multiplicity knots ranging from 0 to 1
knots_x[2 +1:-2 +1] = 0.5                                                 # Why full multiplicity knots in the middle?
knots_x[-2 +1:] = 1

knots_y = torch.zeros(3 + 2 +1)                                 # num_control_points + order == num_control_points + degree + 1                                                           # fill knot vector with full multiplicity knots ranging from 0 to 1
knots_y[2 +1 :-2 +1] = 0.5                                                 # Why full multiplicity knots in the middle?
knots_y[-2 +1:] = 1

nurbs = NURBSSurface(2, 2, 3, 3)
input = (control_points, knots_x.unsqueeze(0), knots_y.unsqueeze(0))
surface_points, surface_normals = nurbs(input)

print(surface_points)
print(surface_normals)
