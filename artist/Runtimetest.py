import time

import torch

def line_plane_intersectionsSOAnew(plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        rays: torch.Tensor,
        surface_points: torch.Tensor,
        epsilon: float = 1e-6,):
    rays_new = rays.view([4, -1])
    ndotu = torch.einsum("ij,ik->j", rays_new, plane_normal).sort()
    #ndotu = ndotu.view([200, -1])
    ndotu_soa_alt = torch.einsum("ijk,jl->ik", rays, plane_normal).view([-1]).sort()
    print(torch.testing.assert_close(ndotu, ndotu_soa_alt))

    return 

def line_plane_intersectionsSOA(plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        rays: torch.Tensor,
        surface_points: torch.Tensor,
        epsilon: float = 1e-6,):
    ndotu = torch.einsum("ijk,jl->ik", rays, plane_normal)
    ds = torch.einsum("ij,ik->j", (plane_point - surface_points) , plane_normal) / ndotu

    return surface_points + torch.einsum("ijk,ik->ijk", rays, ds)

def line_plane_intersectionsAOS(
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        ray_directions: torch.Tensor,
        surface_points: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
    ndotu = ray_directions.matmul(plane_normal)
    ds = (plane_point - surface_points).matmul(plane_normal.to(torch.float)) / ndotu

    return surface_points + ray_directions * ds.unsqueeze(-1)

# Define dimensions
dimension = 4
num_points = 300
num_rays = 200

# Create random tensors with different layouts
planeNormalAOS = torch.randn(dimension)
planePointAOS= torch.randn(dimension)
rayDirectionsAOS = torch.randn(num_rays, num_points, dimension)
rayPointsAOS = torch.randn(num_points, dimension)

planeNormalSOA = torch.randn(dimension, 1)
planePointSOA = torch.randn(dimension, 1)
rayDirectionsSOA = torch.randn(num_rays, dimension, num_points)
rayPointsSOA = torch.randn(dimension, num_points)

# Measure execution time for NxM layout
start_time = time.time()
resultSOAnew = line_plane_intersectionsSOAnew(
    planeNormalSOA, planePointSOA, rayDirectionsSOA, rayPointsSOA
)
elapsed_time_SOAnew = time.time() - start_time

# Measure execution time for NxM layout
start_time = time.time()
resultSOA = line_plane_intersectionsSOA(
    planeNormalSOA, planePointSOA, rayDirectionsSOA, rayPointsSOA
)
elapsed_time_SOA = time.time() - start_time

# Measure execution time for MxN layout
start_time = time.time()
resultAOS = line_plane_intersectionsAOS(
    planeNormalAOS, planePointAOS, rayDirectionsAOS, rayPointsAOS
)
elapsed_time_AOS = time.time() - start_time

print(f"Elapsed time for AOS: {elapsed_time_AOS} seconds")
print(f"Elapsed time for SOA: {elapsed_time_SOA} seconds")
print(f"Elapsed time for SOA: {elapsed_time_SOAnew} seconds")

print(resultAOS.shape, resultSOA.shape)

# Check if the results are similar
# print(torch.allclose(result_NxM, result_MxN))  # Check if the results are similar
