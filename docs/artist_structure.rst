.. _artist_structure:

Understanding ``ARTIST`` Modules: Interactions and User Guidelines
==================================================================

Batch Processing
----------------
``artist`` uses batch processing to parallely align thousands of heliostats and trace millions of rays all at the same time. This is possible because we mainly use tensors to
represent the data in our scenarios. Tensors in ``PyTorch`` are highly efficient when dealing with large data and matrix multiplications, particularly when computing on GPUs.
On the other hand, the same tensor operations on CPUs are significantly slower. This is why we highly recommend running ``artist`` on GPUs. Even on a single GPU, ``artist`` can
compute alignment and raytracing for many heliostats in parallel. To facilitate this we save heliostat and tower data per property, not per object, in large, multidimensional tensors.
For example, heliostats have the following properties:

- ``positions``
- ``surface_points``
- ``surface_normals``
- ``initial_orientations``
- ``kinematic_deviation_parameters``
- ``actuator_parameters``
- ``...``

If we imagine a helisotat field with N=2000 heliostats, there will not be 2000 heliostat-objects, these individual objects do not exist! Instead we have one multidimensional tensor
for each heliostat property, saving property data from each heliostat at specific indices. This results in tensors looking like this:

- ``positions``, where ``torch.Size([N, D])``
- ``surface_points``, where ``torch.Size([N, P, D])``
- ``surface_normals``, where ``torch.Size([N, P, D])``
- ``initial_orientations``, where ``torch.Size([N, D])``
- ``kinematic_deviation_parameters``, where ``torch.Size([N, K])``
- ``actuator_parameters``, where ``torch.Size([N, P, A])``
- ``...``

TODO!!


Heliostat Groups
----------------
In a Solar Tower Power Plant, a heliostat field may consist of multiple types of heliostats designed to optimize sunlight reflection onto a central receiver.
These heliostats can vary in their design. For example, heliostats can be equipped with different amounts of actuators. This variability also allows for the
implementation of various kinematic models to accurately simulate the behavior of the heliostats. The batch processing in ``artist``, to align multiple heliostats
at once requires that all heliostats behave in the same way. With different actuator and kinematic types per heliostat, this is no longer given. This is why ``artist``
internally implements heliostat groups. One ``HeliostatGroup`` includes all heliostats within the heliostat field, that use the same combination of actuator type and
kinematic type. Multiple different groups may exist. Within each group batch processing is possible. The groups are processed sequentially. For the heliostat groups,
actuators and kinemtics, ``artist`` provides abstract base classes. They define common methods implemented by each subtype.
When initializing a ``HeliostatGroup`` in ``artist`` the type of the heliostat group is automatically inferred by checking the provided actuator type and kinematic type.

Actuator Types
^^^^^^^^^^^^^^
Heliostat actuators are motors, responsible for adjusting the orientation of the heliostat's surface to direct sunlight onto a defined aim point. The actuators of heliostats
are described by ``actuator_parameters`` that may contain information on the turning direction of the motors, the step size or offsets. These parameters need to be known for
the initialization. The abstract class ``Actuators`` contains one method to map motor steps to angles and another method to map angles to motor steps. All derived actuator
types override these methods.

A list of supported actuator types:

- ``LinearActuators``
- ``IdealActuators``

If there is a heliostat with a missing actuator type in your heliostat field, you can add another custom class inherting from ``Actuators``.

Kinematic Types
^^^^^^^^^^^^^^^
The kinematic model of the heliostat describes the motion of the it's mechanical system. The kinematic is used to predict the final orientation of the heliostat surface
given variable inputs. In ``artist`` the kinematic is also used to calculate the aligned surface points and normals belonging to a predicted orientation. The choice of kinematic
type can depend on the type of actuators and their amount or the availability of dataset on the positions, orientations, and movements of heliostat mechanical system.
The abstract class ``Kinematic`` contains a method to align the heliostat surface, which internally first computes the desired orientation for an input. All derived kinematic
types override this method.

A list of supported kinematic types:

- ``RigidBody``

If there is a missing kinematic type that suits your heliostats, you can add another custom class inherting from ``Kinematic``.
