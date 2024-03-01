"""This file contains bibliographies for different heliostat prototypes."""

import h5py
from artist import ARTIST_ROOT

# Parameters for an ideal heliostat
_test_heliostat = {
    "height": 4.0,
    "width": 4.0,
    "normal_vectors": [0.0, 0.0, 1.0],
    "disturbance_rotation_angles": [0.0, 0.0, 0.0],
    "facets": {
        "number": 4,
        "positions": [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        "spans_north": [0.0, 1.0, 0.0],
        "spans_east": [-1.0, 0.0, 0.0],
    },
}


#### Dump from binp for test
test_binp_name = "test_data"
_test_binp_heliostat = {
    "height": 2.559999942779541,
    "width": 3.2200000286102295,
    "surface_points": h5py.File(f"{ARTIST_ROOT}/measurement_data/{test_binp_name}.h5", "r")["Points"][()],
    "surface_normals": h5py.File(f"{ARTIST_ROOT}/measurement_data/{test_binp_name}.h5", "r")["Normals"][()],
    "facets": {
        "number": 4,
        "positions": [
            [-0.8075,  0.6425,  0.04019837],
            [0.8075,  0.6425,  0.04019837],
            [-0.8075, -0.6425,  0.04019837],
            [0.8075, -0.6425,  0.04019837],
        ],
        "spans_north": [-0.0,  0.8024845, -0.00498457],
        "spans_east": [6.3749224e-01,  1.9569216e-05,  3.1505227e-03],
    },
}