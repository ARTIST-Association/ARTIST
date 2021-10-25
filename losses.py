# this file is not used right know it inherits all lines of deprecated
# code associated with losses

# loss += loss_func(
#     pred_bitmap,
#     target,
#     lambda rayPoints: compute_receiver_intersections(
#         planeNormal,
#         aimpoint,
#         ray_directions,
#         rayPoints,
#         xi,
#         yi,
#     ),
#     rayPoints,
# )

# loss = th.nn.functional.mse_loss()
# def loss_func(pred, target, compute_intersections, rayPoints):
#     loss = th.nn.functional.mse_loss(pred, target, 0.1)
#     # if cfg.USE_CURL:
#     #     curls = th.stack([
#     #         curl(compute_intersections, rayPoints_)
#     #         for rayPoints_ in rayPoints
#     #     ])
#     #     loss += th.sum(th.abs(curls))
#     return loss
