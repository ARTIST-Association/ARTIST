import torch

def find_spans(number_of_control_points: int,
              degree: int,
              evaluation_points: torch.Tensor,
              knot_vector: torch.Tensor
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
    number_of_control_points : int
        The number of control points.
    degree : int 
        The degree of the NURBS surface in a single direction.
    evaluation_point : torch.Tensor
        The evaluation points.
    knot_vector : torch.Tensor
        Contains all the knots of a single direction.
    
    Returns
    -------
    torch.Tensor
        The knot span index.
    """
    span_indices = torch.empty(len(evaluation_points), dtype=torch.int64)
    
    # Handle special case: the evaluation point is equal to the last knot
    # check whether each evaluation point is not equal to the upper bound of the knot vector
    not_upper_span_indices = evaluation_points != knot_vector[number_of_control_points]
    span_indices[~not_upper_span_indices] = number_of_control_points - 1
    
    # search the span indices
    spans = torch.searchsorted(
        knot_vector,
        evaluation_points[not_upper_span_indices],
        right=True,
    ) - 1
    span_indices[not_upper_span_indices] = spans
    return span_indices

spans = find_spans(3, 2, torch.tensor([1.0000e-05, 3.3334e-01, 6.6666e-01, 9.9999e-01]), torch.tensor([0.0000, 0.0000, 0.1000, 0.3000, 0.6000, 1.0000]))
print(spans)

def nonvanishing_basis_functions(span: torch.Tensor,
                                 evaluation_points: torch.Tensor,
                                 degree: int,
                                 knot_vector: torch.Tensor)->torch.Tensor:
    """
    Compute the nonvanishing basis functions and evaluate them at the given points.

    Calculating each basis function one after the other contains a lot of 
    redundant computations as some parts of one basis function are 
    computed again in the next basis function. That is why in the algorithm 
    below the left and right terms of the basis functions are saved, so that
    they dont have to be recomputed. This makes the algorithm more efficient.
    See `The NURBS Book` page 69-70 for reference.

    Parameters
    ----------
    span : torch.Tensor
        The knot span index.
    evaluation_points : torch.Tensor
        The eveluation_point
    degree : int
        The degree of the NURBS surface in a single direction.
    knot_vector : torch.Tensor
        Contains all the knots of a single direction.
    
    Returns
    -------
    torch.Tensor
        The nonvanishing basis functions applied to the evaluation points.
    """
    basis_values = torch.empty(len(evaluation_points), degree + 1)
    basis_values[:, 0] = 1
    left = torch.empty(basis_values.shape)
    right = torch.empty(basis_values.shape)
    for j in range(1, degree + 1):
        left[:, j] = evaluation_points - knot_vector[span + 1 - j]
        right[:, j] = knot_vector[span + j] - evaluation_points
        saved = torch.zeros(len(evaluation_points))
        for r in range(0, j):
            temp = basis_values[:, r] / (right[:, r + 1] + left[:, j - r])
            basis_values[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        basis_values[:, j] = saved
    return basis_values

    # basis_functions = torch.empty(len(evaluation_points), degree + 1)
    # print(basis_functions.shape)
    # for index, evaluation_point in enumerate(evaluation_points):
    #     for i in range(len(knot_vector)):
    #         if knot_vector[i] <= evaluation_point < knot_vector[i + 1]:
    #             basis_functions[index, 0] = 1
    #         else: 
    #             basis_functions[index, 0] = 0
    #     print(basis_functions)
    #     for p in range(1, degree + 1): 
    #         for i in range(len(knot_vector) - 1):
    #             print(i)
    #             arg1 = ((evaluation_point - knot_vector[i]) / (knot_vector[i + p] - knot_vector[i])) * basis_functions[index, i, p-1]
    #             arg2 = ((knot_vector[i + p + 1] - evaluation_point) / (knot_vector[i + p + 1] - knot_vector[i + 1])) * basis_functions[index, i + 1, p -1]
    #             basis_functions[index, i, p] = arg1 + arg2
        
    # return basis_functions


print(nonvanishing_basis_functions(spans, torch.tensor([1.0000e-05, 3.3334e-01, 6.6666e-01, 9.9999e-01]), 2, torch.tensor([0.0000, 0.0000, 0.1000, 0.3000, 0.6000, 1.0000])))


# def pre_compute_basis_values(evaluation_points_x: torch.Tensor,
#                              evaluation_points_y: torch.Tensor,
#                              knot_vector_x: torch.Tensor,
#                              knot_vector_y: torch.Tensor,
#                              number_of_control_points_x: int,
#                              number_of_control_points_y: int,
#                              degree_x: int,
#                              degree_y: int
# ) -> torch :
#     span_indices_x = find_spans(number_of_control_points_x, degree_x, evaluation_points_x, knot_vector_x)
#     basis_values_x = nonvanishing_basis_functions(span_indices_x, evaluation_points_x, degree_x, knot_vector_x)

#     span_indices_y = find_spans(number_of_control_points_y, degree_y, evaluation_points_y, knot_vector_y)
#     basis_values_y = nonvanishing_basis_functions(span_indices_y, evaluation_points_y, degree_y, knot_vector_y)

#     return span_indices_x, span_indices_y, basis_values_x, basis_values_y


DELTA = 1e-8
class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, 
                 number_of_control_points_x: torch.Tensor,
                 number_of_control_points_y: torch.Tensor,
                 dimension, 
                 degree_x: int,
                 degree_y: int, 
                 knot_vector_x: torch.Tensor, 
                 knot_vector_y: torch.Tensor, 
                 out_dim_u: int=32, 
                 out_dim_v: int=32, 
    ) -> None:
        super(SurfEval, self).__init__()
        self.number_of_control_points_x = number_of_control_points_x
        self.number_of_control_points_y = number_of_control_points_y
        self._dimension = dimension
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.knot_vector_x = knot_vector_x
        self.knot_vector_y = knot_vector_y

        self.evaluation_points_x = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_u, dtype=torch.float32)
        self.evaluation_points_y = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_v, dtype=torch.float32)
        
        self.span_indices_x = find_spans(self.number_of_control_points_x, self.degree_x, self.evaluation_points_x, self.knot_vector_x)
        self.basis_values_x = nonvanishing_basis_functions(self.span_indices_x, self.evaluation_points_x, self.degree_x, self.knot_vector_x)
        self.span_indices_y = find_spans(self.number_of_control_points_y, self.degree_y, self.evaluation_points_y, self.knot_vector_y)
        self.basis_values_y = nonvanishing_basis_functions(self.span_indices_y, self.evaluation_points_y, self.degree_y, self.knot_vector_y)
        

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)

        if self.method == 'cpp':
            out = SurfEvalFunc.apply(input, self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv, self.evaluation_points_x, self.evaluation_points_y, self.number_of_control_points_x, self.number_of_control_points_y, self.degree_x, self.degree_y, self._dimension, self.dvc)
            return out
        elif self.method == 'tc':
            surfaces = (self.Nu_uv[:,0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                input[:,(self.uspan_uv - self.degree_x).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.degree_y).type(torch.LongTensor),:]*\
                self.Nv_uv[:,0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            for r in range(1,self.degree_y+1):
                surfaces += (self.Nu_uv[:,0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                    input[:,(self.uspan_uv - self.degree_x).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.degree_y+r).type(torch.LongTensor),:]*\
                    self.Nv_uv[:,r].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            for l in range(1,self.degree_x+1):
                for r in range(self.degree_y+1):
                    surfaces += (self.Nu_uv[:,l].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                        input[:,(self.uspan_uv - self.degree_x+l).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.degree_y+r).type(torch.LongTensor),:]*\
                        self.Nv_uv[:,r].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            surfaces = surfaces[:,:,:,:self._dimension]/surfaces[:,:,:,self._dimension].unsqueeze(-1)
            return surfaces



class SurfEvalFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension, _device):
        ctx.save_for_backward(ctrl_pts)
        ctx.uspan_uv = uspan_uv
        ctx.vspan_uv = vspan_uv
        ctx.Nu_uv = Nu_uv
        ctx.Nv_uv = Nv_uv
        ctx.u_uv = u_uv
        ctx.v_uv = v_uv
        ctx.m = m
        ctx.n = n
        ctx.p = p
        ctx.q = q
        ctx._dimension = _dimension
        ctx._device = _device

        if _device == 'cuda':
            surfaces = forward(ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        else:
            surfaces = cpp_forward(ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)

        ctx.surfaces=surfaces
        return surfaces[:,:,:,:_dimension]/surfaces[:,:,:,_dimension].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts,  = ctx.saved_tensors
        uspan_uv = ctx.uspan_uv
        vspan_uv = ctx.vspan_uv
        Nu_uv = ctx.Nu_uv
        Nv_uv = ctx.Nv_uv
        u_uv = ctx.u_uv
        v_uv = ctx.v_uv
        m = ctx.m
        n = ctx.n
        p = ctx.p
        q = ctx.q
        _dimension = ctx._dimension
        _device = ctx._device
        surfaces=ctx.surfaces
        grad_sw = torch.zeros((grad_output.size(0),grad_output.size(1),grad_output.size(2),_dimension+1),dtype=torch.float32)
        grad_sw[:,:,:,:_dimension] = grad_output
        if _device == 'cuda':
            grad_sw = grad_sw.cuda()

        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]


        if _device == 'cuda':
            grad_ctrl_pts = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        else:
            grad_ctrl_pts = cpp_backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)

        
        
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None,None