"""This file contains bibliographies for different heliostat prototypes."""

# Parameters for an ideal heliostat
_test_heliostat = {
    "columns": 32,
    "rows": 32,
    "height": 4.0,
    "width": 4.0,
    "position_on_fields": [0.0, 0.0, 0.0],
    "normal_vectors": [0.0, 0.0, 1.0],
    "disturbance_rotation_angles": [0.0, 0.0, 0.0],
    "facets": {
        "number": 4,
        "shape": 0,
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
