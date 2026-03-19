.. _scenario:

The Scenario HDF5 File Format
=============================

In ``ARTIST``, all simulations are based on a so-called *scenario* file. This HDF5 (Hierarchical Data Format) file
contains the complete configuration of a solar tower power plant. It defines, for example, the type, layout, and
orientation of the heliostats in the field, as well as the positions of the receiver and calibration targets.

The HDF5 file format is designed to store and organize large amounts of data in a hierarchical structure. It supports
various data types, including numerical arrays, tables, and metadata, and provides efficient compression and chunking
mechanisms, making it well-suited for handling large datasets. Structurally, HDF5 files are composed of:

- Groups, which act like folders and allow nested organization of data,
- Datasets, storing the actual array-like data, and
- Attributes to attach metadata to the actual data.

They provide a versatile, scalable, and platform-independent way to store, manage, and access large, complex data
structures. That is why they are widely used in research, engineering, and data analysis applications.

Structure of an ``ARTIST`` Scenario
-----------------------------------

An ``ARTIST`` scenario consists of five main elements:

Power Plant
     To provide a realistic geographical context, the power plant location is stored in
     the scenario file. The coordinates are saved in WGS84 coordinates (latitude, longitude, and altitude).
Target Areas
     Every scenario contains at least one target area. A target area is a defined area on the solar tower
     where reflected light is concentrated. If the target area is a receiver, the concentrated light is converted
     into thermal energy for electricity generation or industrial processes. If the target area is a calibration target,
     this area on the tower is used exclusively for calibration tasks, for example in the alignment optimization. A
     ``TargetArea`` object contains information such as position, type, curvature, and other geometric properties.
     The scenario structure supports multiple target areas, which is particularly useful for calibration setups or
     multi-receiver power plants.
Light Sources
     Each scenario contains at least one light source, which models how the incoming radiation is
     generated before being reflected onto a receiver. A light source has a defined type, for example, ``Sun``, and
     specifies how many rays are to be sampled from this light source for ray tracing. The scenario structure allows
     multiple light sources, which can be useful in advanced calibration or testing configurations.
Heliostats
     A concentrating solar power plant relies on mirrors, the heliostats, to reflect light onto
     the receiver. Therefore, an ``ARTIST`` scenario must contain at least one heliostat (typically many). Each
     heliostat includes:

     - A unique ID,
     - a position in the field,
     - an aim point it is focusing on,
     - a surface,
     - a kinematics model, and
     - at least one actuator.

     The surface is the reflective mirror geometry and composed of multiple facets. As each facet is modeled using
     Non-Uniform Rational B-Splines, the corresponding NURBS parameters must be stored in the HDF5 file.

     The kinematics define how the heliostat can move, i.e., the number and orientation of rotation axes and the
     permitted movement directions. Each kinematic configuration contains at least one actuator responsible for
     performing the orientation adjustments.
Prototypes
     In realistic CSP plants, almost all heliostats are identical. Operators typically source heliostats
     from a single manufacturer to reduce maintenance costs and simplify the acquisition process. To avoid redundant
     storage of identical parameters, an ``ARTIST`` scenario includes *prototypes* for the surface,
     kinematics, and actuators. If a heliostat does not specify individual configuration parameters, ``ARTIST``
     automatically loads the corresponding prototype. This significantly reduces file size and improves maintainability.

These five elements result in an ``ARTIST`` scenario HDF5 file with the following structure:

.. code-block:: console

    .
    ├── power_plant [1]
    │   └── position [1,1]
    ├── target_areas [1,*]
    │   ├── receiver1
    │   │   ├── geometry [1,1]
    │   │   ├── position_center [1,1]
    │   │   ├── curvature_e [0,1]
    │   │   ├── curvature_u [0,1]
    │   │   ├── normal_vector [1,1]
    │   │   ├── plane_e [1,1]
    │   │   └── plane_u [1,1]
    │   ├── calibration_target1
    │   │   └── ...
    │   └── ...
    ├── lightsources [1,*]
    │   ├── lightsource1
    │   │   ├── type [1,1]
    │   │   ├── number_of_rays [1,1]
    │   │   └── distribution_parameters [1,1]
    │   │       ├── distribution_type [1,1]
    │   │       ├── mean [0,1]
    │   │       ├── covariance [0,1]
    │   │       └── ...
    │   ├── lightsource2
    │   │   └── ...
    │   └── ...
    ├── heliostats [1,*]
    │   ├── heliostat1
    │   │   ├── id [1,1]
    │   │   ├── position [1,1]
    │   │   ├── surface [0,1]
    │   │   │   └── facets [1,*]
    │   │   │       ├── facet_1
    │   │   │       │   ├── control_points [1,1]
    │   │   │       │   ├── degrees [1,1]
    │   │   │       │   ├── position [1,1]
    │   │   │       │   └── canting [1,1]
    │   │   │       ├── facet_2
    │   │   │       │   └── ...
    │   │   │       └── ...
    │   │   ├── kinematics [0,1]
    │   │   │   ├── type [1,1]
    │   │   │   ├── initial_orientation [1,1]
    │   │   │   └── deviations [0,1]
    │   │   │       ├── first_joint_translation_e [0,1]
    │   │   │       ├── first_joint_tilt_n [0,1]
    │   │   │       └── ...
    │   │   └── actuator [0,*]
    │   │       ├── actuator_1
    │   │       │   ├── type [1,1]
    │   │       │   ├── clockwise_axis_movement [1,1]
    │   │       │   ├── min_max_motor_positions [1,1]
    │   │       │   └── parameters [0,*]
    │   │       │       ├── increment [0,1]
    │   │       │       ├── initial_stroke_length [0,1]
    │   │       │       ├── offset [0,1]
    │   │       │       ├── pivot_radius [0,1]
    │   │       │       └── initial_angle [0,1]
    │   │       └── actuator_2
    │   │           └── ...
    ├── heliostat2
    │   └── ...
    └── ...
    └── prototypes [1,1]
        ├── surface [1,1]
        │   └── facets [1,*]
        │       ├── facet_1
        │       │   ├── control_points [1,1]
        │       │   ├── degrees [1,1]
        │       │   ├── position [1,1]
        │       │   └── canting [1,1]
        │       ├── facet_2
        │       │   └── ...
        │       └── ...
        ├── kinematics [1,1]
        │   ├── type [1,1]
        │   ├── initial_orientation [1,1]
        │   └── deviations [0,1]
        │       ├── first_joint_translation_e [0,1]
        │       ├── first_joint_tilt_e [0,1]
        │       └── ...
        └── actuator [1,*]
            ├── actuator_1
            │   ├── type [1,1]
            │   ├── clockwise_axis_movement [1,1]
            │   ├── min_max_motor_positions [1,1]
            │   └── parameters [0,1]
            │       ├── increment [0,1]
            │       ├── initial_stroke_length [0,1]
            │       ├── offset [0,1]
            │       ├── pivot_radius [0,1]
            │       └── initial_angle [0,1]
            └── actuator_2
                └── ...
