.. _scenario:

The Scenario HDF5 File Format
=============================

In ``ARTIST``, everything is based on a so-called *scenario* file. In this HDF5 (Hierarchical Data Format) file, the
complete setup of the solar tower power plant, such as the type, layout, and orientation of the heliostats in the field
or the positions of the receiver and calibration targets, are defined. The HDF5 file format is designed to store and
organize large amounts of data in a hierarchical structure. It supports various data types, including numerical arrays,
tables, and metadata, and provides efficient compression and chunking techniques, making them suitable for handling large
datasets. HDF5 files are composed of groups and datasets. Groups act like folders, allowing data to
be organized in a nested manner, while datasets store the actual data in array-like structures. HDF5 files can also
include attributes to store metadata alongside the data. They provide a versatile, scalable, and platform-independent
way to store, manage, and access complex data structures, which is why they are widely used in research, engineering,
and data analysis applications.

The Scenario structure of ``ARTIST`` has five main elements:

   - **Power Plant:** To integrate the the scenario into a realistic environment, the power plant location is contained
     in the scenario. The coordinates are saved in WGS84 coordinates including latitude, longitude and altitude.
   - **Target Areas:** Every scenario in ``ARTIST`` contains at least one target area. A target area is the area on the solar tower
     where the light is concentrated. If the target area is a receiver the resulting heat energy is used to generate
     electricity or for industrial processes. If the target area is a calibration target, this area on the tower is used
     for calibration purposes only, for example in the alignment optimization. A ``TargetArea`` object contains information
     such as the position, the type, whether the area is curved or not etc. ``ARTIST`` makes use of a scenario structure
     that allows for multiple target areas, which is useful when considering calibration for multiple targets or for large
     power plants with multiple receivers.
   - **Light Sources:** An ``ARTIST`` scenario also contains at least one light source. A light source models how the
     light, which is eventually reflected onto a receiver, is generated. A light source must have a type, for example, a
     ``Sun``, and defines how many rays are to be sampled from this light source for ray tracing. Since it may be
     interesting to model multiple light sources for calibration purposes, the ``ARTIST`` scenario structure also
     supports more than one light source.
   - **Heliostats:** A Concentrating Solar Power Plant (CSP) relies on mirrors, so-called *heliostats*, to reflect the
     light onto the receiver. Therefore, an ``ARTIST`` scenario must contain at least one (and usually multiple)
     heliostats. Besides its unique ID, its position, and the aim point it is focusing on, a heliostat
     requires a *surface*, a *kinematics*, and at least one *actuator*. The surface is the reflective surface of the
     heliostat and is made up of multiple *facets*. Each of these facets is modeled by Non-Uniform Rational B-Splines
     (NURBS) and, therefore, the parameters required to load the NURBS must be defined in the HDF5 file. The kinematics
     defines how the heliostat can be orientated, i.e., where are the axes of rotation, how many of these axes exist and
     what directions of movement are allowed. Each kinematics also contains at least one actuator which is responsible
     for performing the orientation changes.
   - **Prototypes:** Typically, *almost all heliostats are identical* for a realistic CSP. This
     is because CSP operators often source their heliostats from one manufacturer to reduce maintenance costs and
     simplify the acquisition process. In such a setting, it would be inefficient to save identical heliostat parameters
     for every heliostat in the scenario. Therefore, an ``ARTIST`` scenario also contains *prototypes* for the surface,
     kinematics, and actuators. If an individual heliostat does not have any individual configuration parameters in the
     scenario, then ``ARTIST`` will automatically load the heliostat prototype.

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
