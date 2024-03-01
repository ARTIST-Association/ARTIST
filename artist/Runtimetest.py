import torch
import time

def line_plane_intersectionsNxM(
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        rayDirections: torch.Tensor,
        rayPoints: torch.Tensor,
        epsilon: float = 1e-6, ):
        ndotu = planeNormal.matmul(rayDirections)
        ds = (planeNormal.to(torch.float)).matmul(planePoint - rayPoints) / ndotu

        return planePoint + rayPoints * ds

def line_plane_intersections(
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        rayDirections: torch.Tensor,
        rayPoints: torch.Tensor,
        epsilon: float = 1e-6, ):
        ndotu = rayDirections.matmul(planeNormal)
        ds = (planePoint - rayPoints).matmul(planeNormal.to(torch.float)) / ndotu

        return planePoint + rayPoints * ds
# Define dimensions
N = 3
M = 2000000

# Create random tensors with different layouts
planeNormalNxM = torch.randn(1, N)
planePointNxM= torch.randn(N, 1)
rayDirections_NxM = torch.randn(N, M)
rayPoints_NxM = torch.randn(N, M)

planeNormalMxN = planeNormalNxM.t().contiguous()
planePointMxN = planePointNxM.t().contiguous()
rayDirections_MxN = rayDirections_NxM.t().contiguous()
rayPoints_MxN = rayPoints_NxM.t().contiguous()

# Measure execution time for NxM layout
start_time = time.time()
result_NxM = line_plane_intersectionsNxM(planeNormalNxM, planePointNxM, rayDirections_NxM, rayPoints_NxM)
elapsed_time_NxM = time.time() - start_time

# Measure execution time for MxN layout
start_time = time.time()
result_MxN = line_plane_intersections(planeNormalMxN, planePointMxN, rayDirections_MxN, rayPoints_MxN)
elapsed_time_MxN = time.time() - start_time

print(f"Elapsed time for NxM: {elapsed_time_NxM} seconds")
print(f"Elapsed time for MxN: {elapsed_time_MxN} seconds")

print(result_MxN.shape, result_NxM.shape)

# Check if the results are similar
#print(torch.allclose(result_NxM, result_MxN))  # Check if the results are similar
