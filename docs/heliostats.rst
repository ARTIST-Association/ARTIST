.. _heliostats:

Understanding Heliostats
========================

``ARTIST`` is a digital twin for concentrating solar tower power plants. One of the most important aspects of these plants
are the **heliostats** - the mirrors that reflect light onto the receiver. Heliostats aren't just perfect flat mirrors; they are
complex structures that must be accurately modeled. In ``ARTIST``, we model heliostats using three key components: a
**surface**, a **kinematic model**, and **actuators**. This page provides a brief overview of how we include heliostats
in ``ARTIST`` and details the specific models we support.

Surfaces
^^^^^^^^
The surface is a crucial element of a heliostat, as it's responsible for reflecting light onto the receiver. Accurately
modeling the surface is of utmost importance. In ``ARTIST``, a surface consists of multiple facets, which can be canted
at an angle to improve sunlight concentration.

Most commonly, heliostats in ``ARTIST`` consist of four facets, as shown in the image below.

.. figure:: ./images/Facet_Properties.pdf
   :alt: Facet Properties Overview
   :width: 55%
   :align: center

As you can see, each facet has a ``position`` relative to the heliostat's center and canting direction vectors
(``canting_e``, ``canting_n``) that define its orientation.

Kinematic
^^^^^^^^^
The heliostat's kinematic model describes the motion of its mechanical system. It's used to predict the final orientation
of the heliostat surface based on variable inputs. The kinematic model is also used to calculate the aligned surface points
and normals for a predicted orientation. The choice of kinematic type can depend on the type and number of actuators or
the availability of a dataset on the positions, orientations, and movements of the heliostat's mechanical system.

The abstract class ``Kinematic`` contains a method to align the heliostat surface, which internally first computes the
desired orientation for a given input. All derived kinematic types override this method.

``ARTIST`` currently supports the following kinematic type:

- ``RigidBody``

This rigid body kinematic model uses a two-actuator structure, allowing movement in two directions. These actuators
introduce mechanical offsets, described by translation vectors for three components: joint one, joint two, and the concentrator.
These vectors point in the east, north, and up directions, as shown in the image below.

.. figure:: ./images/kinematic_translation.pdf
   :alt: Kinematic Translations Overview
   :width: 55%
   :align: center

In ``ARTIST`` these translations, as well as possible rotations of each joint, are included in the ``translation_deviation_parameters``
and ``rotation_deviation_parameters`` tensors (see :py:class:`artist.field.kinematic_rigid_body.RigidBody`), allowing multiple rigid body
kinematics with slightly different translations and rotations to be effectively modeled.
Joint one and two and the concentrator all have translations in the east north and up direction, making the ``translation_deviation_parameters``
a 9-dimensional tensor. Only joint one and two have ``rotation_deviation_parameters``. Joint one has these deviations only in the north and
up direction, whereas joint two has them only in east and north, making this a 4-dimensional parameter for the rigid body kinematic.

Actuators
^^^^^^^^^
Heliostat actuators are the motors responsible for adjusting the heliostat's surface orientation to direct sunlight onto
a defined aim point. The actuators are described by the actuator parameters, which may contain information on motor
turning direction, step size, or offsets. The actuator parameters are split into ``non_optimizable_parameters`` and ``optimizable_parameters``
when initializing the actuators. The abstract class ``Actuators`` contains one method to map motor steps to angles and another
to map angles to motor steps. All derived actuator types override these methods.

``ARTIST`` currently supports the following actuator types:

- ``LinearActuators``
- ``IdealActuators``

The ``LinearActuator`` is modeled on the actuator used in the Jülich power plant and includes the following parameters:

.. figure:: ./images/Actuator_properties.pdf
   :alt: Actuator Properties
   :width: 100%
   :align: center

.. list-table:: Linear Actuator Non-Optimizable Parameters
   :header-rows: 1
   :widths: 20 80

   * - Parameter Name
     - Description
   * - type
     - The type of actuator.
   * - clockwise_axis_movement
     - A boolean indicating if the movement direction is clockwise.
   * - min_motor_pos
     - The smallest motor position the actuator accepts.
   * - max_motor_pos
     - The largest motor position the actuator accepts.
   * - increment
     - The total number of increments per full stroke.
   * - offset
     - The physical offset from the actuator axis to the pivot. (2) in the image above.
   * - pivot_radius
     - The radius from the pivot center to the actuator anchor. (1) in the image above.


.. list-table:: Linear Actuator Optimizable Parameters
   :header-rows: 1
   :widths: 20 80

   * - Parameter Name
     - Description
   * - initial_angle
     - The starting angular position of the actuator.
   * - initial_stroke_length
     - The initial extension length of the actuator. (3) in the image above.


In ``ARTIST`` the linear actuator is physics-informed to prevent invalid values and NaNs during forward and backward
passes. The parameters increment, initial_stroke_length, offset and pivot_radius are defined strictly positive using
the ``softplus`` function, ensuring physically meaningful values throughout the optimizations. Additionally, the law of
cosines, is used in both methods of the linear actuator to compute angles or motor steps. This involves the stroke
length, offset, and pivot radius of each linear actuator. Together they make up a triangle, defining how far the actuator
is from the actual rotational center of the joint. To make the law of cosines applicable, the sides (offset and pivot
radius) of the triangle must satisfy the triangle inequality:

.. math::

    |offset - pivot\_radius| \leq stroke\_length \leq offset + pivot\_radius

This ensures that a valid triangle can always be formed. With this physics-informed backbone, the linear actuator can
accurately model real actuator behavior in an efficient manner.


The ``IdealActuator`` has no optimizable parameters. The non-optimizable parameters only include the ``type``, ``clockwise_axis_movement``,
``min_motor_pos`` and ``max_motor_pos``.
