import torch

class NURBSSurface(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
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
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        control_points, knot_vector_x, knot_vector_y = input
        control_points = control_points.unsqueeze(dim=0)

        # normalize knot vector
        knot_vector_x_ = torch.cumsum(torch.where(knot_vector_x<0.0, knot_vector_x*0+1e-4, knot_vector_x), dim=1)
        knot_vector_x = (knot_vector_x_ - knot_vector_x_[:, 0].unsqueeze(-1)) / (knot_vector_x_[:, -1].unsqueeze(-1) - knot_vector_x_[:, 0].unsqueeze(-1))
        knot_vector_y_ = torch.cumsum(torch.where(knot_vector_y<0.0, knot_vector_y*0+1e-4, knot_vector_y), dim=1)
        knot_vector_y = (knot_vector_y_ - knot_vector_y_[:, 0].unsqueeze(-1)) / (knot_vector_y_[:, -1].unsqueeze(-1) - knot_vector_y_[:, 0].unsqueeze(-1))

        # find span indices (based on A2.1) 
        evaluation_points_x = self.evaluation_points_x.unsqueeze(0)
        span_indices_x = torch.stack([torch.min(torch.where((evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1))>1e-8, evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1), (evaluation_points_x - knot_vector_x[s,self.degree_x:-self.degree_x].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.degree_x for s in range(knot_vector_x.size(0))])
        evaluation_points_x = self.evaluation_points_x.squeeze(0)

        # find basis values (based on A2.2)
        basis_values = [evaluation_points_x*0 for i in range(self.degree_x+1)]
        basis_values[0] = evaluation_points_x*0 + 1
        for k in range(1,self.degree_x+1):
            saved = (evaluation_points_x)*0.0
            for r in range(k):
                left = torch.stack([knot_vector_x[s,span_indices_x[s,:] + r + 1] for s in range(knot_vector_x.size(0))])
                right = torch.stack([knot_vector_x[s,span_indices_x[s,:] + 1 - k + r] for s in range(knot_vector_x.size(0))])
                temp = basis_values[r]/((left - evaluation_points_x) + (evaluation_points_x - right))
                temp = torch.where(((left - evaluation_points_x) + (evaluation_points_x - right))==0.0, evaluation_points_x*0+1e-4, temp)
                basis_values[r] = saved + (left - evaluation_points_x)*temp
                saved = (evaluation_points_x - right)*temp
            basis_values[k] = saved

        basis_values_x = torch.stack(basis_values).permute(1,0,2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        
        # find span indices (based on A2.1) 
        evaluation_points_y = self.evaluation_points_y.unsqueeze(0)
        span_indices_y = torch.stack([torch.min(torch.where((evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1))>1e-8, evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1), (evaluation_points_y - knot_vector_y[s,self.degree_y:-self.degree_y].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.degree_y for s in range(knot_vector_y.size(0))])
        evaluation_points_y = self.evaluation_points_y.squeeze(0)
        
        # find basis values (based on A2.2)
        basis_values = [evaluation_points_y*0 for i in range(self.degree_y+1)]
        basis_values[0] = evaluation_points_y*0 + 1
        for k in range(1,self.degree_y+1):
            saved = (evaluation_points_y)*0.0
            for r in range(k):
                left = torch.stack([knot_vector_y[s,span_indices_y[s,:] + r + 1] for s in range(knot_vector_y.size(0))])
                right = torch.stack([knot_vector_y[s,span_indices_y[s,:] + 1 - k + r] for s in range(knot_vector_y.size(0))])
                temp = basis_values[r]/((left - evaluation_points_y) + (evaluation_points_y - right))
                temp = torch.where(((left - evaluation_points_y) + (evaluation_points_y - right))==0.0, evaluation_points_y*0+1e-4, temp)
                basis_values[r] = saved + (left - evaluation_points_y)*temp
                saved = (evaluation_points_y - right)*temp
            basis_values[k] = saved

        basis_values_y = torch.stack(basis_values).permute(1,0,2).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)
        
        # added parentheses to make it work
        pts = torch.stack([
            torch.stack([
                torch.stack([
                    control_points[s,(span_indices_x[s,:]-(self.degree_x+l)),:,:][:,(span_indices_y[s,:]-(self.degree_y+r)),:]
                    for r in range(self.degree_y+1)
                ])
                for l in range(self.degree_x+1)
            ]) 
            for s in range(knot_vector_x.size(0))
        ])

        surfaces = torch.sum((basis_values_x*pts)*basis_values_y, (1,2))
        surfaces = surfaces[:,:,:,:self._dimension]
        return surfaces



# class BasisFunc(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, u, U, uspan_uv, p):
#         ctx.save_for_backward(U)
#         ctx.uspan_uv = uspan_uv
#         ctx.u = u
#         ctx.p = p

#         u = u.squeeze(0)
#         Ni = [u*0 for i in range(p+1)]
#         Ni[0] = u*0 + 1
#         for k in range(1,p+1):
#             saved = (u)*0.0
#             for r in range(k):
#                 UList1 = torch.stack([U[s, uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
#                 UList2 = torch.stack([U[s, uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
#                 temp = Ni[r]/((UList1 - u) + (u - UList2))
#                 temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-8, temp)
#                 Ni[r] = saved + (UList1 - u)*temp
#                 saved = (u - UList2)*temp
#             Ni[k] = saved

#         Nu_uv = torch.stack(Ni).permute(1,0,2)
#         ctx.Nu_uv = Nu_uv
#         return Nu_uv

#     @staticmethod
#     def backward(ctx, grad_output):
#         U = ctx.saved_tensors[0]
#         uspan_uv = ctx.uspan_uv
#         p = ctx.p
#         Nu_uv = ctx.Nu_uv
#         u = ctx.u

#         UList = torch.stack([U[s, uspan_uv[s,:]] for s in range(U.size(0))])

#         dNi = [grad_output[:,0,:]*0 for i in range(p+1)]
#         dNi[0] = grad_output[:,0,:]*0 + 0.5*UList*Nu_uv[:,0,:]

#         for k in reversed(range(1,p+1)):
#             tempdNi = dNi[k]*grad_output[:,k,:]
#             for r in reversed(range(k)):
#                 UList1 = torch.stack([U[s, uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
#                 UList2 = torch.stack([U[s, uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
#                 tempdNi = tempdNi*(u-UList2)
#                 dNi[r] += (UList1 - u)*tempdNi
#                 dNi[r] = dNi[r]/((UList1 - u) + (u - UList2))

#         dNu_uv = torch.stack(dNi).permute(1,0,2)

#         dU = U*0
#         for s in range(U.size(0)):
#             for k in range(0,p+1):
#                 dU[s, :].scatter_(-1, (uspan_uv[s,:] + k).type_as(uspan_uv), dNu_uv[s, k, :], reduce='add')
#         dU = dU*U


#         # for s in range(U.size(0)):
#         #     for t in range(uspan_uv.size(1)):
#         #         for k in range(1,p+1):
#         #             tempdU = dU*0
#         #             for r in range(k):
#         #                 tempdU[s, uspan_uv[s,t] + r] += grad_output[s, r, t]*U[s, uspan_uv[s,t] + r + 1]
#         #                 tempdU[s, uspan_uv[s,:] + r + 1] += (-1)*grad_output[s, 1 - k + r, t]*U[s, uspan_uv[s,:] + 1 - k + r]
#         #             dU += tempdU

#         # dU = U*0
#         # for s in range(U.size(0)):
#         #     for k in range(0,p+1):
#         #         dU[s, uspan_uv[s,:]] += grad_output[s, k, :]*Nu_uv[s, k, :]*((100/1.00)**2)

#         return Variable(U*0), Variable(dU), Variable(U*0), None
    

control_points_shape = (4, 5)                        
control_points = torch.empty(control_points_shape + (3,))   
knots_x = torch.tensor([0, 0, 0.1, 0.2, 0.3, 1]).unsqueeze(dim=0)
knots_y = torch.tensor([0, 0, 0.1, 0.2, 0.3, 1]).unsqueeze(dim=0)
nurbs = NURBSSurface(2, 2, 3, 3)
input = (control_points, knots_x, knots_y)
output = nurbs(input)
print(output)