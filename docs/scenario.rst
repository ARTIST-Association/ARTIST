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
     this area on the tower is used exclusively for calibration tasks, for example in the alignment optimization.
     There are currently two types of target areas implemented in ``ARTIST``, which differ in their geometry: planar and
     cylindrical. Planar target areas are commonly used for calibration surfaces, while cylindrical target areas represent
     receiver geometries. Due to these geometric differences, each target area type requires its own ray-tracing methods.
     ``ARTIST`` handles this automatically by sorting the target areas provided in the scenario HDF5 files into the appropriate
     classes and applying the correct methods for each type. The scenario structure supports multiple target areas,
     which is particularly useful for calibration setups or multi-receiver power plants, where different target area types
     may be used within a single simulation. Each target area includes basic information such as its name, position, and normal vector.
     In addition, planar target areas store geometric dimensions, while cylindrical target areas include parameters such as
     radius, axis, height, and opening angle.
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
    ├── target_areas_planar [1,*]
    │   ├── calibration_target1
    │   │   ├── position_center [1,1]
    │   │   ├── normal_vector [1,1]
    │   │   ├── plane_e [1,1]
    │   │   └── plane_u [1,1]
    │   ├── calibration_target2
    │   │   └── ...
    │   └── ...
    ├── target_areas_cylindrical [1,*]
    │   ├── receiver1
    │   │   ├── cylinder_axis [1,1]
    │   │   ├── cylinder_center [1,1]
    │   │   ├── cylinder_height [1,1]
    │   │   ├── cylinder_opening_angle [1,1]
    │   │   ├── cylinder_radius [1,1]
    │   │   └── cylinder_normal [1,1]
    │   ├── receiver2
    │   │   └── ...
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
    ├── heliostats [1,1]                         (attribute: layout = "columnar")
    │   ├── names [N]
    │   ├── id [N]
    │   ├── position [N,4]
    │   ├── surface [0,1]
    │   │   ├── heliostat_indices [N_s]
    │   │   ├── control_points [N_s,F,U,V,3]
    │   │   ├── degrees [N_s,F,2]
    │   │   ├── position [N_s,F,4]
    │   │   └── canting [N_s,F,2,4]
    │   ├── kinematics [0,1]
    │   │   ├── heliostat_indices [N_k]
    │   │   ├── type [N_k]
    │   │   ├── initial_orientation [N_k,4]
    │   │   └── deviations [0,1]
    │   │       ├── first_joint_translation_e [N_k]
    │   │       ├── first_joint_tilt_n [N_k]
    │   │       └── ...
    │   └── actuator [0,1]
    │       ├── heliostat_indices [N_a]
    │       ├── names [N_a,A]
    │       ├── type [N_a,A]
    │       ├── clockwise_axis_movement [N_a,A]
    │       ├── min_max_motor_positions [N_a,A,2]
    │       └── parameters [0,1]
    │           ├── increment [N_a,A]
    │           ├── initial_stroke_length [N_a,A]
    │           ├── offset [N_a,A]
    │           ├── pivot_radius [N_a,A]
    │           └── initial_angle [N_a,A]
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

The heliostats are stored *column by column*: each heliostat parameter is a single dataset holding the values of all
heliostats, with the heliostats along the first dimension. This keeps the number of HDF5 objects independent of the
number of heliostats, so that even fields with thousands of heliostats are saved and loaded within seconds. Here,
``N`` is the number of heliostats, ``N_s``, ``N_k``, and ``N_a`` are the numbers of heliostats with an individual
surface, kinematics, and actuators, ``F`` is the number of facets, ``U`` and ``V`` are the numbers of control points
per facet, and ``A`` is the number of actuators per heliostat.

- Heliostats are sorted by name, and all ``[N]`` datasets follow this order.
- Individual surfaces, kinematics, and actuators are only stored for the heliostats that have them.
  ``heliostat_indices`` lists the rows of these heliostats in ``names``; all other heliostats use the prototypes.
- Missing kinematics deviations and actuator parameters are stored as ``NaN`` and default to zero when loading.
- All individual surfaces must have the same shape, and all heliostats with individual actuators must have the same
  number of actuators. If this is not the case, the ``H5ScenarioGenerator`` warns and stores the heliostats in the
  legacy layout described below.

.. note::

    **Legacy heliostat layout (ARTIST 2.0.0 and earlier).** Scenarios created with ``ARTIST`` 2.0.0 or earlier store
    one HDF5 group per heliostat instead, with every parameter as a separate dataset. All other parts of the scenario
    are identical:

    .. code-block:: console

        heliostats [1,*]
        ├── heliostat1
        │   ├── id [1,1]
        │   ├── position [1,1]
        │   ├── surface [0,1]
        │   │   └── facets [1,*]
        │   │       ├── facet_1
        │   │       │   ├── control_points [1,1]
        │   │       │   ├── degrees [1,1]
        │   │       │   ├── position [1,1]
        │   │       │   └── canting [1,1]
        │   │       └── ...
        │   ├── kinematics [0,1]
        │   │   ├── type [1,1]
        │   │   ├── initial_orientation [1,1]
        │   │   └── deviations [0,1]
        │   │       └── ...
        │   └── actuator [0,*]
        │       ├── actuator_1
        │       │   ├── type [1,1]
        │       │   ├── clockwise_axis_movement [1,1]
        │       │   ├── min_max_motor_positions [1,1]
        │       │   └── parameters [0,*]
        │       │       └── ...
        │       └── ...
        ├── heliostat2
        │   └── ...
        └── ...

    ``ARTIST`` still loads these scenarios and yields exactly the same scenario as from the columnar layout, but logs
    a warning, as loading is much slower for large heliostat fields. Existing scenarios can be converted without
    regenerating them:

    .. code-block:: python

        from artist.io import convert_scenario_to_columnar

        convert_scenario_to_columnar("scenario.h5", "scenario_columnar.h5")

    ``ARTIST`` 2.0.0 and earlier cannot load scenarios in the columnar layout.
