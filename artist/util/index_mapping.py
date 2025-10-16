actuator_type = 0
"""Index for the actuator type."""
actuator_clockwise_movement = 1
"""Index for the actuator clockwise movement."""
actuator_min_motor_position = 2
"""Index for the actuator minimum motor position."""
actuator_max_motor_position = 3
"""Index for the actuator maximum motor position."""
actuator_increment = 4
"""Index for the actuator increment."""
actuator_offset = 5
"""Index for the actuator offset."""
actuator_pivot_radius = 6
"""Index for the actuator pivot radius."""
actuator_initial_angle = 0
"""Index for the actuator initial angle."""
actuator_initial_stroke_length = 1
"""Index for the actuator initial stroke length."""
first_joint_translation_e = 0
"""Index for the first joint translation in the east direction."""
first_joint_translation_n = 1
"""Index for the first joint translation in the north direction."""
first_joint_translation_u = 2
"""Index for the first joint translation in the up direction."""
second_joint_translation_e = 3
"""Index for the second joint translation in the east direction."""
second_joint_translation_n = 4
"""Index for the second joint translation in the north direction."""
second_joint_translation_u = 5
"""Index for the second joint translation in the up direction."""
concentrator_translation_e = 6
"""Index for the concentrator translation in the east direction."""
concentrator_translation_n = 7
"""Index for the concentrator translation in the north direction."""
concentrator_translation_u = 8
"""Index for the concentrator translation in the up direction."""
first_joint_tilt_n = 0
"""Index for the first joint tilt in the north direction."""
first_joint_tilt_u = 1
"""Index for the first joint tilt in the up direction."""
second_joint_tilt_e = 2
"""Index for the second joint tilt in the east direction."""
second_joint_tilt_n = 3
"""Index for the second joint tilt in the north direction."""
data_actuator_min_motor_position = 0
"""Index for the actuator minimum motor position when loading data."""
data_actuator_max_motor_position = 1
"""Index for the actuator maximum motor position when loading data."""
facet_canting_e = 0
"""Index for the facet canting in the east direction."""
facet_canting_n = 1
"""Index for the facet canting in the north direction."""
paint_actuator_type = 0
"""Index for the paint actuator type."""
paint_actuator_clockwise_axis_movement = 1
"""Index for the paint actuator clockwise axis movement."""
paint_actuator_min_max_motor_positions = 2
"""Index for the paint actuator minimum motor positions."""
paint_actuator_parameters = 3
"""Index for the paint actuator parameters."""
stral_surface_header_start = 5
"""Index for the stral surface header start."""
stral_surface_header_end = 7
"""Index for the stral surface header end."""
stral_n_xy_start = 0
"""Index for the stral number of facets calculation start."""
stral_n_xy_end = 1
"""Index for the stral number of facets calculation end."""
stral_facet_start = 1
"""Index for the stral facet header start."""
stral_facet_end = 4
"""Index for the stral facet header end."""
stral_canting_1 = 0
"""Index for the stral canting 1."""
stral_canting_1_start = 4
"""Index for the first stral canting start."""
stral_canting_1_end = 7
"""Index for the first stral canting end."""
stral_canting_2 = 1
"""Index for the stral canting 2."""
stral_canting_2_start = 7
"""Index for the second stral canting start."""
stral_canting_2_end = 10
"""Index for the second stral canting end."""
stral_number_of_points = 10
"""Index for the stral number of points."""
stral_surface_points_start = 0
"""Index for the stral surface points start."""
stral_surface_points_end = 3
"""Index for the stral surface points end."""
stral_surface_normals_start = 3
"""Index for the stral surface normals start."""
stral_surface_normals_end = 6
"""Index for the stral surface normals end."""
first_facet = 0
"""Index for the first facet of a heliostat."""
h5_control_points_u = 0
"""Index for the control points in the u dimension in a h5 file."""
h5_control_points_v = 1
"""Index for the control points in the v dimension in a h5 file."""
surface_points_from_tuple = 0
"""Index for surface points in 2D tuple."""
surface_normals_from_tuple = 1
"""Index for surface normals in 2D tuple."""
nurbs_surfaces = 0
"""Index to access nurbs surfaces."""
nurbs_facets = 1
"""Index to access nurbs facets."""
nurbs_u = 0
"""Index to access the nurbs u parametric direction."""
nurbs_v = 1
"""Index to access the nurbs v parametric direction."""
nurbs_control_points_start = 2
"""Index to access the control points."""
nurbs_knots_unbatched = 0
"""Index to access the nurbs knots in unbatched tensors."""
nurbs_knots_batched = 2
"""Index to access the nurbs knots in batched tensors."""
nurbs_span_lower = 0
"""Index to access the lower spans."""
nurbs_span_upper = 1
"""Index to access the upper spans."""
nurbs_evaluation_points = 2
"""Index to access the nurbs evaluation points."""
nurbs_spans = 3
"""Index to access the nurbs spans."""
nurbs_ndu_basis_i = 0
"""Index to access nurbs basis function index in recursion."""
nurbs_ndu_basis_j = 1
"""Index to access the nurbs basis function recursion order."""
nurbs_ndu_basis_index_0 = 0
"""Index for the zeroth basis function index."""
basis_function_derivative_order = 0
"""Index to access the basis function derivative order."""
nurbs_control_points_u = 2
"""Index to access the nurbs u direction in the control points."""
nurbs_control_points_v = 3
"""Index to access the nurbs v direction in the control points."""
nurbs_control_points = 4
"""Index to access the nurbs control points."""
nurbs_derivative_order_0 = 0
"""Index to access the derivative order of zero."""
nurbs_derivative_order_1 = 1
"""Index to access the derivative order of one."""
nurbs_normals = 3
"""Index to access the surface normals from within the nurbs."""