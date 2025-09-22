.. _artist_under_hood:

``ARTIST``: What's Happening Under The Hood?
============================================

We designed ``ARTIST`` to be robust, work effectively in parallel, and take advantage of GPU acceleration. While most of
these design decisions are handled automatically, a few points are worth noting. This page provides a brief overview of
some of the core principles governing how ``ARTIST`` handles processes and data internally.

Coordinates
-----------

``ARTIST`` uses the east, north, up (ENU) coordinate system in a **four-dimensional** format. To understand the
implications of this, consider two example tensors:

.. code-block:: console

    point_tensor = torch.tensor([e, n, u, 1])
    direction_tensor = torch.tensor([e, n, u, 0])

Both tensors are similar in their first three elements:

* The first element is the **east** coordinate.
* The second element is the **north** coordinate.
* The third element is the **up** coordinate.

However, the fourth element is an extension to a **4D** representation of **3D** coordinates. This enables ``ARTIST`` to
perform *rotations* and *translations* within a single *affine transformation matrix*, which improves efficiency.
With this **4D** representation, it's important to understand:

* The final element in a tensor representing a point **is always 1**.
* The final element in a tensor representing a direction **is always 0**.

Batch Processing
----------------

``ARTIST`` uses batch processing to align thousands of heliostats and trace millions of rays in parallel. This is
possible because we use tensors to represent the data in our scenarios. Tensors in ``PyTorch`` are highly efficient when
dealing with large data and matrix multiplications, particularly when computing on GPUs. By contrast, the same tensor
operations on CPUs are significantly slower. This is why we highly recommend running ``ARTIST`` on GPUs. Even on a single
GPU, ``ARTIST`` can compute alignment and raytracing for many heliostats in parallel.

To facilitate this, we save heliostat and tower data internally by property, not by object, in large, multidimensional
tensors. For example, some heliostat properties are:

* ``positions``
* ``surface_points``
* ``surface_normals``
* ``initial_orientations``
* ``kinematic_deviation_parameters``
* ``actuator_parameters``
* ...

If we consider a heliostat field with N=2000 heliostats, there won't be 2000 heliostat objects. Instead of these
individual objects, we have one multidimensional tensor for each heliostat property, saving property data from each
heliostat at specific indices. This results in the following important tensors:

.. list-table:: Important ``ARTIST`` Tensors and Shapes
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Shape
     - Description
   * - ``positions``
     - ``torch.Size([N, D])``
     - Positions of all heliostats in the group.
   * - ``surface_points``
     - ``torch.Size([N, P, D])``
     - Surface points of all heliostats.
   * - ``surface_normals``
     - ``torch.Size([N, P, D])``
     - Surface normals of all heliostats.
   * - ``initial_orientations``
     - ``torch.Size([N, D])``
     - Initial orientations of all heliostats.
   * - ``kinematic_deviation_parameters``
     - ``torch.Size([N, K])``
     - Kinematic deviation parameters for each heliostat.
   * - ``actuator_parameters``
     - ``torch.Size([N, A_param, A_num])``
     - Actuator parameters for each heliostat.
   * - ``nurbs_control_points``
     - ``torch.Size([N, F, u, v, 3])``
     - Control points for NURBS surfaces for all heliostats.
   * - ``nurbs_degrees``
     - ``torch.Size([2])``
     - Spline degrees for NURBS surfaces in the u and v directions.
   * - ``active_heliostats_mask``
     - ``torch.Size([N])``
     - A boolean mask indicating which heliostats are active.
   * - ``active_surface_points``
     - ``torch.Size([N_active, P, D])``
     - Surface points of all active heliostats.
   * - ``active_surface_normals``
     - ``torch.Size([N_active, P, D])``
     - Surface normals of all active heliostats.
   * - ``active_nurbs_control_points``
     - ``torch.Size([N_active, F, u, v, 3])``
     - NURBS control points for all active heliostats.
   * - ``preferred_reflection_directions``
     - ``torch.Size([N_active, P, D])``
     - Preferred reflection directions for all active heliostats.

with:

.. list-table:: Explanation of the Tensor Shapes
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``N``
     - The total number of heliostats in the group.
   * - ``D``
     - The number of dimensions, which is always 4 in ARTIST, representing a 4D coordinate system.
   * - ``P``
     - The number of surface points (or surface normals) per heliostat.
   * - ``K``
     - The number of kinematic parameters.
   * - ``F``
     - The number of facets per heliostat.
   * - ``u``
     - The number of control points in the u-direction for NURBS surfaces (see :ref:`our tutorial on NURBS <nurbs>`).
   * - ``v``
     - The number of control points in the v-direction for NURBS surfaces (see :ref:`our tutorial on NURBS <nurbs>`).
   * - ``A_param``
     - The number of actuator parameters for this actuator type.
   * - ``A_num``
     - The number of actuators for the selected kinematic type.
   * - ``N_active``
     - The number of active heliostats.

Note that since a heliostat's surface is modeled by multiple facets (see :ref:`this info on heliostats <heliostats>`),
the number of surface points is internally divided among these facets. Additionally, for raytracing, we always consider
each surface point to have a single surface normal, and therefore the number of surface points is always equal to the
number of surface normals.

What may be confusing is the ``N_active`` parameter, which refers to active heliostats. The ``N_active`` parameter exists
because it is possible to only address certain heliostats during operational tasks. It is also possible, that ``N_active``
is larger than ``N``. This occurs during calibration or optimization tasks, when a single heliostat may be duplicated
multiple times, to account for multiple training data samples. ``N_active`` sums all duplicates of all activated
heliostats. To better understand this, we
need to consider heliostat groups, which we discuss in the next section.

Heliostat Groups
----------------

In a Solar Tower Power Plant, a heliostat field may consist of multiple types of heliostats with varying designs. For
example, heliostats can be equipped with different numbers of actuators or varying kinematic models. The batch processing
in ``ARTIST``, which processes multiple heliostats at once, requires that all heliostats behave in the same way. This is
not the case with different actuator and kinematic types per heliostat.

This is why ``ARTIST`` internally implements heliostat groups. A single ``HeliostatGroup`` includes all heliostats
within the field that use the same combination of actuator and kinematic types. Multiple different groups may exist.
Within each group, batch processing is possible, and the groups are processed sequentially. For the heliostat groups,
actuators, and kinematics, ``ARTIST`` provides abstract base classes that define common methods implemented by each
subtype.

When initializing a ``HeliostatGroup`` in ``ARTIST``, the type of the heliostat group is automatically inferred by
checking the provided actuator and kinematic types. To summarize: you should never have to worry about creating a
heliostat group yourself; they exist and are handled automatically!
