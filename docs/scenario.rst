.. scenario:

The Scenario HDF5 File Format
=============================

In ``ARTIST``, the complete setup of the thermal solar power plant, such as the type, layout, and orientation of the
heliostats in the field or the position of the receiver, is defined in a so-called *scenario HDF5 file*. The HDF
(Hierarchical Data Format) file format is designed to store and organize large amounts of data in a hierarchical
structure. It supports various data types, including numerical arrays, images, tables, and metadata, and provides
efficient compression and chunking techniques, making them suitable for handling large datasets. HDF files work by
organizing data into a hierarchical structure composed of groups and datasets. Groups act like folders, allowing data to
be organized in a nested manner, while datasets store the actual data in array-like structures. HDF files can also
include attributes to store metadata alongside the data. They provide a versatile, scalable, and platform-independent
way to store, manage, and access complex data structures, which is why they are widely used in research, engineering,
and data analysis applications.

.. code-block:: console

        .
    ├── receivers [1,*]
    │   ├── receiver1
    │   │   ├── type [1,1] # e.g. planar
    │   │   ├── position_center [1,1]
    │   │   ├── curvature_e [0,1]
    │   │   ├── curvature_u [0,1]
    │   │   ├── normal_vector [1,1]
    │   │   ├── plane_e [1,1]
    │   │   ├── plane_u [1,1]
    │   │   ├── resolution_e [1,1]
    │   │   └── resolution_u [1,1]
    │   ├── receiver2
    │   │   └── ...
    │   └── ...
    ├── lightsources [1,*]
    │   ├── lightsource1
    │   │   ├── number_of_rays [1,1]
    │   │   └── distribution_parameters [1,1]
    │   │       ├── distribution_type [1,1]
    │   │       ├── mean [0,1]
    │   │       ├── variance [0,1]
    │   │       └── ...
    │   ├── lightsource2
    │   │   └── ...
    │   └── ...
    ├── heliostats [1,*]
    │   ├── heliostat1
    │   │   ├── id [1,1]
    │   │   ├── position [1,1]
    │   │   ├── aim_point [1,1]
    │   │   ├── surface [0,1]
    │   │   │   └── facets [1,*]
    │   │   │       ├── facet1
    │   │   │       │   ├── control_points_e [1,1]
    │   │   │       │   ├── control_points_u [1,1]
    │   │   │       │   ├── knots_e [1,1]
    │   │   │       │   ├── knots_u [1,1]
    │   │   │       │   ├── width [1,1]
    │   │   │       │   ├── height [1,1]
    │   │   │       │   ├── position [1,1]
    │   │   │       │   ├── canting_e [1,1]
    │   │   │       │   └── canting_n [1,1]
    │   │   │       ├── facet2
    │   │   │       │   └── ...
    │   │   │       └── ...
    │   │   ├── kinematic [0,1]
    │   │   │   ├── type [1,1]
    │   │   │   ├── offsets [0,3]
    │   │   │   │   ├── offset_e
    │   │   │   │   ├── offset_n
    │   │   │   │   └── offset_u
    │   │   │   └── deviations [0,*]
    │   │   │       ├── first_joint_translation [1,1]
    │   │   │       ├── first_joint_tilt_e [1,1]
    │   │   │       └── ...
    │   │   └── actuator [0,1]
    │   │       ├── type [1,1]
    │   │       └── deviations [0,*]
    │   │           ├── first_joint_increment [1,1]
    │   │           ├── first_joint_radius [1,1]
    │   │           └── ...
    │   ├── heliostat2
    │   │   └── ...
    │   └── ...
    └── prototypes [1,1]
        ├── surface [1,1]
        │   └── facets [1,*]
        │       ├── facet1
        │       │   ├── control_points_e [1,1]
        │       │   ├── control_points_u [1,1]
        │       │   ├── knots_e [1,1]
        │       │   ├── knots_u [1,1]
        │       │   ├── width [1,1]
        │       │   ├── height [1,1]
        │       │   ├── position [1,1]
        │       │   ├── canting_e [1,1]
        │       │   └── canting_n [1,1]
        │       ├── facet2
        │       │   └── ...
        │       └── ...
        ├── kinematic [1,1]
        │   ├── type [1,1]
        │   ├── offsets [0,3]
        │   │   ├── offset_e
        │   │   ├── offset_n
        │   │   └── offset_u
        │   └── deviations [0,*]
        │       ├── first_joint_translation [1,1]
        │       ├── first_joint_tilt_e [1,1]
        │       └── ...
        └── actuator [1,1]
            ├── type [1,1]
            └── deviations [0,*]
                ├── first_joint_increment [1,1]
                ├── first_joint_radius [1,1]
                └── ...
