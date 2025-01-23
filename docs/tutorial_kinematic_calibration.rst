.. _tutorial_kinematic_calibration:

``ARTIST`` Tutorial: Kinematic Calibration
==========================================

.. note::

    You can find the corresponding ``Python`` script for this tutorial here:
    https://github.com/ARTIST-Association/ARTIST/blob/main/tutorials/02_alignment_optimization_tutorial.py

This tutorial provides a brief introduction to ``ARTIST`` showcasing how the kinematic calibration is performed.
The tutorial will run through some basic concepts necessary to understanding ``ARTIST`` including:

- Why do we need to calibrate the kinematic.
- How to load calibration data.
- How to set up the ``AlignmentOptimizer`` responsible for the kinematic calibration.

It is best if you already know about the following processes in ``ARTIST``

- How to load a scenario.
- Activating the kinematic in a heliostat to align this heliostat for raytracing.
- Performing heliostat raytracing to generate a flux density image on the receiver.

If you need help with this look into our other tutorials such as the tutorial on :ref:`heliostat raytracing <tutorial_heliostat_raytracing>`.

Kinematic Calibration Basics
----------------------------
In the real world most components of the kinematic have mechanical errors. This means if we tell an actuator to orient
a heliostat along a specific angle, the heliostat might not end up pointing exactly at the specified aim point.
In ``ARTIST`` we create a digital twin of a solar tower power plant. In the computer simulation, the heliostat will, per default,
point exactly where we tell it to. To keep the predictions made with ``ARTIST`` as accurate as possible we need to
consider the mechanical errors and offsets of the real-world kinematic. In the kinemactic calibration process the kinematic module
learns all offset or deviation parameters of the real-world kinematic, to mimic its behavior.
Calibrating the kinematic in ``ARTIST`` requires calibration data, the flux density distributions used for the calibration
can be gained by pointing single heliostats at calibration targets.

Loading the Calibration Data
----------------------------
Calibration data can be accessed from the ``PAINT`` database: https://paint-database.org/.
To load the calibration data into the ``ARTIST`` environment we use the ``paint_loader`` and specify
the ``calibration_properties_path`` by providing a valid path and the ``power_plant_position``, so that coordinates
can be transformed into the local ENU coordinate system used in ``ARTIST``.

.. code-block::

    # Load the calibration data.
    (
        calibration_target_name,
        center_calibration_image,
        incident_ray_direction,
        motor_positions,
    ) = paint_loader.extract_paint_calibration_data(
        calibration_properties_path=calibration_properties_path,
        power_plant_position=example_scenario.power_plant_position,
        device=device,
    )

Now we have all the calibration data we need to calibrate the kinematic of the provided scenario.

Optimizable parameters
----------------------
In the pre-generated scenario used in this tutorial, we use a rigid body kinematic. For this kinematic type
there are alltogether 28 optimizable parameters. You can select all of them with the following code:

.. code-block::

    # Get optimizable parameters. This will select all 28 kinematic parameters.
    parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(
        kinematic=example_scenario.heliostats.heliostat_list[0].kinematic
    )

Setting up the ``AlignmentOptimizer``
-------------------------------------
The alignment optimizer object is responsible for the kinematic calibration. We define the light source by
creating an ``AlignmentOptimizer`` object as shown below:

.. code-block::

    # Create alignment optimizer.
    alignment_optimizer = AlignmentOptimizer(
        scenario=example_scenario,
        optimizer=optimizer,
        scheduler=scheduler,
    )

This object defines the following alignment optimizer properties:

- The ``scenario`` provides all of the environment variables.
- The ``optimizer`` is a ``torch.optim.Optimizer`` like ``torch.optim.Adam`` that contains the optimizable parameters.
- The ``scheduler`` is a ``torch.optim.lr_scheduler`` like ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

Optimizing the parameters
-------------------------
The set up is now complete and the kinematic calibration can begin. The kinematic calibration is an optimization process.
We start the optimization process by calling:

.. code-block::

    optimized_parameters, optimized_scenario = alignment_optimizer.optimize(
        tolerance=tolerance,
        max_epoch=max_epoch,
        center_calibration_image=center_calibration_image,
        incident_ray_direction=incident_ray_direction,
        motor_positions=motor_positions,
        device=device,
    )

Currently there are two methods to calibrate the kinematic. Either we use geometric considerations and the
motor positions from the calibration data or we optimize using flux density distributions and the differentiable
raytracer. The kinematic calibration via the motor position is generally faster and produces better results in less
time. However, choosing the optimization method depends on the available calibration data. Both methods
need information about:

- The center of the measured flux density distribution,
- The incident ray direction during the measurement,

The more efficient calibration via the ``motor_positions`` additionally needs information about the motor positions
that were measured during the data acquisition. The ``motor_positions`` is an optional parameter in the ``optimize()``
function above. Since we included them here, the calibration happens via the motor positions.

Optimization methods
--------------------
Here is the workflow of the kinematic calibration with motor positions:

- We start with default values for all optimizable paramters.
- We calculate the preferred reflection direction of our heliostat through knowledge about the
  center of the calibration flux density distribution.
- In the optimization loop we calculate the current orientation of the heliostat from the motor positions,
  then we calculate the actual reflection direction of the heliostat. The loss is defined by the
  difference between the actual reflection direction and the preferred reflection direction from the calibration data.
- The optimizer updates the optimizable parametrs until it is accurate enough or the maximum number of epochs is reached.

Here is the workflow of the kinematic calibration with the differentiable raytracer.

- We start with default values for all optimizable paramters.
- In the optimization loop we align the heliostat by providing the incident ray direction of the calibration data.
  Then we create the Heliostat Raytracer by specifying the used calibration target instead of the revceiver. We trace the rays
  and create a bitmap of the flux density distribution. From this distribution we calculate the center. The loss is defined as the
  difference between the actual center from the raytraced distribution and the center of the calibration data.
- The optimizer updates the optimizable parametrs until it is accurate enough or the maximum number of epochs is reached.

Both optimization methods return the optimized parameters and the optimized scenario that is ready to be used for raytracing.
