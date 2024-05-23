.. _scenario:

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

The Scenario structure of ``ARTIST`` has four main elements:

   - **Receivers:** Every scenario in ``ARTIST`` contains at least one receiver. The receiver is the object where the
     light is concentrated and resulting heat energy used to generate electricity or for industrial processes. A
     ``Receiver`` object contains information such as the position, the type, whether the receiver is curved or not etc.
     Since large Concentrating Solar Power Plants (CSPs) may contain multiple receiver, ``ARTIST`` is modelled on a
     scenario structure that also allows multiple receiver.
   - **Light Sources:** An ``ARTIST`` scenario also contains at least one light source. A light source models how the
     light is generated, which is eventually reflected onto a receiver. A light source must have a type, for example, a
     ``Sun``, and defines how many rays are to be sampled from this light source for raytracing. Since it may be
     interesting to model multiple light sources for calibration purposes, the ``ARTIST`` scenario structure also
     supports more than one light source.
   - **Heliostats:** A Concentrating Solar Power Plant (CSP) relies on mirrors, so-called *heliostats* to reflect the
     light on to the receiver. Therefore, an ``ARTIST`` scenario must contain at least one (and usually multiple)
     heliostats. As well as a unique ID, a position, and the aim point the heliostat is focusing on, a heliostat
     requires a *surface*, a *kinematic*, and at least one *actuator*. The surface is the reflective surface of the
     heliostat and is made up of multiple *facets*. Each of these facets is modelled by Non-Uniform Rational B-Splines
     (NURBS) and, therefore, the parameters required to load the NURBS must be defined in the HDF5 file.
     These surfaces are often not ideal, and as a result, NURBS are required to learn the minute deformations. The
     second important heliostat attribute is the kinematic, which defines how the heliostat can be orientated, i.e.
     where are the axes of rotation, how many of these axes exist and what directions of movement are allowed. Each
     kinematic also contains at least one actuator which is responsible for performing the orientation.
   - **Prototypes:** If we are considering a realistic CSP then typically, *almost all heliostats are identical!*. This
     is because CSP operators often source their heliostats from one manufacturer to reduce maintenance costs and
     simplify the acquisition process. In such a setting, it would be inefficient to save identical heliostat parameters
     for every heliostat in the scenario. Therefore, an ``ARTIST`` scenario also contains *prototypes* for the surface,
     kinematic, and actuators. If an individual heliostat does not have any individual configuration parameters in the
     scenario, then ``ARTIST`` will automatically load the heliostat prototype.

These four elements result in an ``ARTIST`` scenario HDF5 file with the following structure:

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
    │   │   ├── type [1,1]
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
    │   │   │       │   ├── control_points [1,1]
    │   │   │       │   ├── degree_e [1,1]
    │   │   │       │   ├── degree_n [1,1]
    │   │   │       │   ├── number_of_eval_points_e [1,1]
    │   │   │       │   ├── number_of_eval_points_n [1,1]
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
    │   │   │       ├── first_joint_translation [0,1]
    │   │   │       ├── first_joint_tilt_e [0,1]
    │   │   │       └── ...
    │   │   └── actuators [0,*]
    │   │       ├── actuator1
    │   │       │   ├── type [1,1]
    │   │       │   └── parameters [0,*]
    │   │       │       ├── first_joint_increment [0,1]
    │   │       │       ├── first_joint_radius [0,1]
    │   │       │       └── ...
    │   │       └── actuator2
    │   │           └── ...
    │   ├── heliostat2
    │   │   └── ...
    │   └── ...
    └── prototypes [1,1]
        ├── surface [1,1]
        │   └── facets [1,*]
        │       ├── facet1
        │       │   ├── control_points [1,1]
        │       │   ├── degree_e [1,1]
        │       │   ├── degree_n [1,1]
        │       │   ├── number_of_eval_points_e [1,1]
        │       │   ├── number_of_eval_points_n [1,1]
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
        │       ├── first_joint_translation [0,1]
        │       ├── first_joint_tilt_e [0,1]
        │       └── ...
        └── actuators [1,*]
            ├── actuator1
            │   ├── type [1,1]
            │   └── parameters [0,*]
            │       ├── first_joint_increment [0,1]
            │       ├── first_joint_radius [0,1]
            │       └── ...
            └── actuator2
                └── ...
