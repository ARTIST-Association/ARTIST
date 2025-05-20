.. _tutorial_heliostat_raytracing:

``ARTIST`` Tutorial: Heliostat Ray Tracing
=========================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/01_heliostat_raytracing_tutorial.py

This tutorial provides a brief introduction to ``ARTIST`` showcasing how Heliostat Ray Tracing is performed. The tutorial
will run through some basic concepts necessary to understanding ``ARTIST`` including:

- How to load a scenario.
- How to select specific helisotats for alignment and raytracing.
- Activating the kinematic in the heliostats to align the heliostats for ray tracing.
- Performing heliostat ray tracing to generate flux density images on the target areas on the tower.

Loading a Scenario
------------------
Before we load the scenario, you need to decide which scenario file to use. This tutorial is based on a simple scenario
which you can create in the tutorial on :ref:`generating a scenario <tutorial_generating_scenario>`. However, since generating a
scenario involves training Non-Uniform Rational B-Splines (NURBS) and may take a while, we have also provided some
scenario files in the artist tutorials directory.

Please adjust the path and name of the ``scenario_file`` variable:

.. code-block::

    # Specify the path to your scenario.h5 file.
    scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

Once you have adjusted this parameter, you can load a scenario in ``ARTIST`` by simply calling the
``load_scenario_from_hdf5()`` method, which is a ``Python`` ``classmethod`` that initializes a ``Scenario`` object based on
the configuration contained in the HDF5 file:

.. code-block::

    # Load the scenario.
    with h5py.File(scenario_path) as scenario_path:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_path, device=device
        )

When loading the scenario, a large number of log messages is generated:

.. code-block::

    [2025-03-10 11:40:25,108][artist.util.scenario][INFO] - Loading an ``ARTIST`` scenario HDF5 file. This scenario file is version 1.0.
    [2025-03-10 11:40:25,127][artist.field.tower_target_area_array][INFO] - Loading the tower target area array from an HDF5 file.
    [2025-03-10 11:40:25,127][artist.field.tower_target_area][INFO] - Loading receiver from an HDF5 file.
    [2025-03-10 11:40:25,951][artist.field.tower_target_area][WARNING] - No curvature in the east direction set for the receiver!
    [2025-03-10 11:40:25,951][artist.field.tower_target_area][WARNING] - No curvature in the up direction set for the receiver!
    [2025-03-10 11:40:25,951][artist.scene.light_source_array][INFO] - Loading a light source array from an HDF5 file.
    [2025-03-10 11:40:25,952][artist.scene.sun][INFO] - Loading sun_1 from an HDF5 file.
    [2025-03-10 11:40:25,952][artist.scene.sun][INFO] - Initializing a sun modeled with a multivariate normal distribution.
    [2025-03-10 11:40:27,644][artist.util.scenario][WARNING] - No individual kinematic first_joint_translation_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic first_joint_translation_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic first_joint_translation_u for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic first_joint_tilt_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic first_joint_tilt_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic first_joint_tilt_u for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_translation_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_translation_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_translation_u for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_tilt_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_tilt_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic second_joint_tilt_u for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_translation_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_translation_u for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_translation_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_tilt_e for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_tilt_n for None set. Using default values!
    [2025-03-10 11:40:27,645][artist.util.scenario][WARNING] - No individual kinematic concentrator_tilt_u for None set. Using default values!
    [2025-03-10 11:40:27,647][artist.util.scenario][WARNING] - No individual increment set for actuator_1. Using default values!
    [2025-03-10 11:40:27,647][artist.util.scenario][WARNING] - No individual initial_stroke_length set for actuator_1 on None. Using default values!
    [2025-03-10 11:40:27,647][artist.util.scenario][WARNING] - No individual offset set for actuator_1 on None. Using default values!
    [2025-03-10 11:40:27,647][artist.util.scenario][WARNING] - No individual pivot_radius set for actuator_1 on None. Using default values!
    [2025-03-10 11:40:27,647][artist.util.scenario][WARNING] - No individual initial_angle set for actuator_1 on None. Using default values!
    [2025-03-10 11:40:27,648][artist.util.scenario][WARNING] - No individual increment set for actuator_2. Using default values!
    [2025-03-10 11:40:27,648][artist.util.scenario][WARNING] - No individual initial_stroke_length set for actuator_2 on None. Using default values!
    [2025-03-10 11:40:27,648][artist.util.scenario][WARNING] - No individual offset set for actuator_2 on None. Using default values!
    [2025-03-10 11:40:27,648][artist.util.scenario][WARNING] - No individual pivot_radius set for actuator_2 on None. Using default values!
    [2025-03-10 11:40:27,648][artist.util.scenario][WARNING] - No individual initial_angle set for actuator_2 on None. Using default values!
    [2025-03-10 11:40:27,648][artist.field.heliostat_field][INFO] - Loading a heliostat field from an HDF5 file.
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_translation_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_translation_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_translation_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_tilt_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_tilt_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic first_joint_tilt_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,677][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_translation_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_translation_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_translation_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_tilt_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_tilt_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic second_joint_tilt_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_translation_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_translation_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_translation_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_tilt_e for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_tilt_n for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,678][artist.field.heliostat_field][WARNING] - No individual kinematic concentrator_tilt_u for heliostat_1 set. Using default values!
    [2025-03-10 11:40:27,679][artist.field.heliostat_field][WARNING] - No individual increment set for actuator_1. Using default values!
    [2025-03-10 11:40:27,679][artist.field.heliostat_field][WARNING] - No individual initial_stroke_length set for actuator_1 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,679][artist.field.heliostat_field][WARNING] - No individual offset set for actuator_1 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,679][artist.field.heliostat_field][WARNING] - No individual pivot_radius set for actuator_1 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,679][artist.field.heliostat_field][WARNING] - No individual initial_angle set for actuator_1 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,680][artist.field.heliostat_field][WARNING] - No individual increment set for actuator_2. Using default values!
    [2025-03-10 11:40:27,680][artist.field.heliostat_field][WARNING] - No individual initial_stroke_length set for actuator_2 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,680][artist.field.heliostat_field][WARNING] - No individual offset set for actuator_2 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,680][artist.field.heliostat_field][WARNING] - No individual pivot_radius set for actuator_2 on heliostat_1. Using default values!
    [2025-03-10 11:40:27,680][artist.field.heliostat_field][WARNING] - No individual initial_angle set for actuator_2 on heliostat_1. Using default values!

These log messages consist of three brackets:

   - The first bracket, e.g., ``[2025-03-10 11:40:25,108]``, displays the time stamp.
   - The second bracket, e.g., ``[artist.util.scenario]``, displays the file that generated the log message.
   - The third bracket, e.g., ``[INFO]`` or ``[WARNING]``, displays the level for which the log message is being generated.
   - Finally, after the three brackets, the log message is printed.

Whilst there are quite a few log messages, there are two important aspects you should note:

   1. The majority of the messages are warnings – however, this is not a problem. We are considering a simplistic
      scenario, and as a result do not include specific kinematic or actuator parameters or deviations. Therefore,
      ``ARTIST`` automatically uses the default values. In this case, this is the desired behavior, and we can ignore the
      warnings!
   2. The remaining messages are info messages. These messages are informing us of the names of the objects being
      loaded from the HDF5 file, important information about these objects, and at the very end stating that the
      heliostat does not contain individual parameters and is (as we expect) being loaded using the prototypes.

Before we start using this scenario, we can inspect it, for example by printing the scenario properties or investigating
what type of light source and target area is included:

.. code-block::

    # Inspect the scenario.
    print(scenario)
    print(f"The light source is a {scenario.light_sources.light_source_list[0]}.")
    print(f"The first target area is a {scenario.target_areas.names[0]}.")
    print(
        f"The first heliostat in the first group in the field is heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]}."
    )
    print(
        f"Heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]} is located at: {scenario.heliostat_field.heliostat_groups[0].positions[0].tolist()}."
    )
    print(
        f"Heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]} is aiming at: {scenario.heliostat_field.heliostat_groups[0].kinematic.aim_points[0].tolist()}."
    )

This code generates the following output:

.. code-block::

    ARTIST Scenario containing:
            A Power Plant located at: [0.0, 0.0, 0.0] with 1 Target Area(s), 1 Light Source(s), and 1 Heliostat(s).
    The light source is a Sun().
    The first target area is a receiver.
    The first heliostat in the first group in the field is heliostat heliostat_1.
    Heliostat heliostat_1 is located at: [0.0, 5.0, 0.0, 1.0].
    Heliostat heliostat_1 is aiming at: [0.0, -50.0, 0.0, 1.0].


Selecting Active Heliostats and Target Areas
--------------------------------------------
In ARTIST the information about the helisotats is saved per heliostat property. There is one tensor containing
all heliostat positions from a specific heliostat group. Similarly there is one tensor containing all aim points and so on.
To address a specific heliostat, it is important to know its index. To activate one or more heliostats for the
alignment process or raytracing, you can specify these indices in the ``active_heliostats_indices`` tensor, like this:

.. code-block::
    # We will choose the first Heliostat, with index 0 by activating it.
    active_heliostats_indices = torch.tensor([0], device=device)

The same is true for the target areas.

.. code-block::
    # We select the first target area as the designated target for this heliostat.
    target_area_indices = torch.tensor([0], device=device)


Aligning Heliostats
--------------------
Before we can start ray tracing, we need to align the heliostats. In the current scenario, our heliostats are
initialized pointing straight up at the sky. Unfortunately, this orientation is not very useful for reflecting
sunlight from the sun onto the receiver that is located in the south (see aim point above).

Therefore, we make use of our knowledge regarding the:

- Position of the heliostats,
- Aim points, and
- Kinematic model,

to align the heliostats in an optimal position for reflection. To perform this orientation, we need an *incident ray
direction*, i.e., a direction vector, originating in the light source position and pointing towards the heliostat field.
``ARTIST`` can accomodate heliostats with various differnt kinematic and actuator types. Since each kinematic type and
actuator type computes the orientations of aligned heliostats slightly different, we need to seperate the heliostats into
``HeliostatGroup`` groups. ``ARTIST`` handels this automatically.
Given an *incident ray direction*, we can align the heliostats with the following code:

.. code-block::

    # Align the heliostat(s).
    scenario.heliostat_field.heliostat_groups[
        0
    ].align_surfaces_with_incident_ray_directions(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_indices=active_heliostats_indices,
        device=device,
    )

We can compare the original surface and the aligned surface of the first heliostat in the heliostat field
in the following plot:

.. figure:: ./images/tutorial_surface.png
   :width: 100 %
   :align: center

Since both the target area (receiver) and the sun are directly to the south of the heliostat field, this alignment is completely plausible.
The heliostat is rotated 90 degrees along the east axis to reflect the sunlight back in the direction it is coming
from.

Ray Tracing
----------
With the heliostats now aligned, it is time to perform some ray tracing to generate flux density images.

In this tutorial, we are considering *heliostat ray tracing*. Heliostat ray tracing (as it's name suggests) traces rays
of sunlight from the heliostat. If we were to trace rays from the sun, then only a small portion would hit the heliostat
and even a smaller portion of these rays would hit the receiver. Therefore, heliostat ray tracing can be computationally
efficient. Concretely, the heliostat ray tracing involves three main steps:

1. We calculate the preferred reflection directions of all heliostats. This preferred reflection direction models the direction of a ray
   coming directly from the sun to the heliostats, i.e., along the incident ray direction. Specifically, we reflect this
   ray at every point on the heliostats to generate multiple *ideal* reflections.
2. This single ray only models an *ideal* direction, but we need to account for all possible rays coming from the sun.
   Therefore, we use our model of the sun to create *distortions* which we then use to slightly alter the preferred
   reflection directions multiple times, thus generating many realistically reflected rays.
3. We trace these rays onto the target area by performing a *line-plane intersection* and determining the resulting flux
   density image on the receiver.

Luckily, ``ARTIST`` automatically performs all of these steps within the ``HeliostatRayTracer`` class! Therefore, ray tracing
with ``ARTIST`` involves two simple lines of code. First, we define the ``HeliostatRayTracer``. A ``HeliostatRayTracer``
only requires a ``Scenario`` object as an argument and the specification of which ``HelisotatGroup`` is currently regarded.

.. code-block::

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=scenario.heliostat_field.heliostat_groups[0],
    )

Internally, a ``HeliostatRayTracer`` uses a ``torch.Dataset`` to generate rays and the distortion of the preferred
reflection directions, line plane intersections, and calculation of the resulting flux density images. This process
runs parallel for all heliostats in the scenario. It is further possible to use a data-parallel setup for the ``HeliostatRayTracer``
to split the computation along multiple devices. See the tutorial on :ref:`distributed raytracing. <tutorial_distributed_raytracing>

With everything now set up, we can generate a flux density image by calling the ``trace_rays()`` function with the
desired incident ray directions, the active heliostat indices and the target area indices (for this tutorial we use the receiver).

.. code-block::

    # Perform heliostat-based ray tracing.
    image_south = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_indices=active_heliostats_indices,
        target_area_indices=target_area_indices,
        device=device,
    )

If we plot the output, we get the following flux density image!

.. figure:: ./images/tutorial_south_flux.png
   :width: 80 %
   :align: center

That's it – a simple example of heliostat ray tracing with ``ARTIST``!

Of course, this one scenario is capable of performing ray tracing for any incident ray direction. For example, we can consider
three further incident ray directions and perform ray tracing using a helper function that combines alignment and
ray tracing with the following code:

.. code-block::

    # Define light directions.
    incident_ray_direction_east = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=device)
    incident_ray_direction_west = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    incident_ray_direction_above = torch.tensor([[0.0, 0.0, -1.0, 0.0]], device=device)

    # Perform alignment and ray tracing to generate flux density images.
    image_east = align_and_trace_rays(
        light_direction=incident_ray_direction_east,
        active_heliostats_indices=active_heliostats_indices,
        target_area_indices=target_area_indices,
        device=device,
    )
    image_west = align_and_trace_rays(
        light_direction=incident_ray_direction_west,
        active_heliostats_indices=active_heliostats_indices,
        target_area_indices=target_area_indices,
        device=device,
    )
    image_above = align_and_trace_rays(
        light_direction=incident_ray_direction_above,
        active_heliostats_indices=active_heliostats_indices,
        target_area_indices=target_area_indices,
        device=device,
    )

If we were to now plot the results of all four considered incident ray directions, we get the following image:

.. figure:: ./images/tutorial_multiple_flux.png
   :width: 100 %
   :align: center
